import os
import random
import logging
import argparse
import json
from datetime import datetime
from dataclasses import asdict


import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree

from config import Config
from models import Generator, Discriminator
from train import train, evaluate
from utils import visualize_graphs, analyze_dataset_statistics, get_target_distribution_stats, extract_individual_graphs, compute_graph_statistics, compute_mmd, wl_graph_hash

# --------------------------
# Logging
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------
#  Data Collection Helpers
# --------------------------
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_graph_structures(graphs, filename, logger):
    """
    Saves a list of PyG Data objects to a JSON file in a parseable format.
    """
    data_list = []
    for i, g in enumerate(graphs):
        label = int(g.y.item()) if g.y is not None else -1
        edges = []
        if g.edge_index is not None and g.edge_index.size(1) > 0:
            edge_tensor = g.edge_index.t().cpu().numpy()
            edges = edge_tensor.tolist()

        graph_data = {
            "id": i,
            "label": label,
            "num_nodes": int(g.x.size(0)),
            "num_edges": len(edges),
            "edges": edges
        }
        data_list.append(graph_data)

    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f, indent=None, cls=NumpyEncoder)
        logger.info(f"Saved {len(graphs)} graph structures to {filename}")
    except Exception as e:
        logger.error(f"Failed to save graph structures: {e}")

def compute_and_save_detailed_stats(graphs, config, filename, logger):
    """
    Computes detailed statistics (Degree, Clustering, Spectral, Node/Edge counts)
    separated by class label and saves to JSON.
    """
    if not graphs:
        logger.warning("No graphs provided for statistics calculation.")
        return

    graphs_by_class = {}
    for g in graphs:
        label = int(g.y.item()) if g.y is not None else -1
        if label not in graphs_by_class:
            graphs_by_class[label] = []
        graphs_by_class[label].append(g)

    output_stats = {}

    for label, class_graphs in graphs_by_class.items():
        if not class_graphs:
            continue
            
        num_nodes = [g.x.size(0) for g in class_graphs]
        num_edges = [g.edge_index.size(1) / 2.0 for g in class_graphs]
        
        basic_stats = {
            "avg_nodes": float(np.mean(num_nodes)),
            "std_nodes": float(np.std(num_nodes)),
            "avg_edges": float(np.mean(num_edges)),
            "std_edges": float(np.std(num_edges)),
            "count": len(class_graphs)
        }

        current_max_deg = 0
        for g in class_graphs:
             if g.edge_index.numel() > 0:
                 d = degree(g.edge_index[0], g.x.size(0))
                 m = d.max().item()
                 if m > current_max_deg:
                     current_max_deg = m
        
        deg_bin_width = current_max_deg / config.num_bins if current_max_deg > 0 else 0.0
        clus_bin_width = 1.0 / config.num_bins
        spec_bin_width = 2.0 / config.num_bins

        raw_stats = compute_graph_statistics(class_graphs, num_bins=config.num_bins)
        
        avg_degree_dist = np.mean(raw_stats['degrees'], axis=0).tolist()
        avg_clustering_dist = np.mean(raw_stats['clustering'], axis=0).tolist()
        avg_spectral_dist = np.mean(raw_stats['spectral'], axis=0).tolist()

        output_stats[label] = {
            "basic": basic_stats,
            "distribution_params": {
                "degree": { "max_val": int(current_max_deg), "bin_width": deg_bin_width },
                "clustering": { "max_val": 1.0, "bin_width": clus_bin_width },
                "spectral": { "max_val": 2.0, "bin_width": spec_bin_width }
            },
            "distributions": {
                "degree": avg_degree_dist,
                "clustering": avg_clustering_dist,
                "spectral": avg_spectral_dist
            }
        }

    try:
        with open(filename, 'w') as f:
            json.dump(output_stats, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Saved detailed statistics to {filename}")
    except Exception as e:
        logger.error(f"Failed to save statistics: {e}")


# --------------------------
#  Main Function
# --------------------------
def main():
    # 1. PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description="Graph Generation Experiment (In-Training GA)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--no-refine", action="store_true", help="Disable GA refinement entirely")
    args = parser.parse_args()

    config = Config()

    if args.seed is not None:
        config.seed = args.seed
    if args.dataset is not None:
        config.dataset_name = args.dataset
    if args.no_refine:
        config.use_refinement = False
    
    # 2. CREATE DYNAMIC DIRECTORY STRUCTURE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refine_tag = "STE" if config.use_refinement else "NoRefine"
    run_name = f"{timestamp}_Seed{config.seed}_{refine_tag}"

    config.results_dir = os.path.join(config.results_dir, config.dataset_name, run_name)
    config.save_dir = os.path.join(config.save_dir, config.dataset_name, run_name)

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    params_file = os.path.join(config.results_dir, "parameters.txt")
    with open(params_file, "w") as f:
        json.dump(asdict(config), f, indent=4)
    logger.info(f"Configuration saved to {params_file}")

    log_file = os.path.join(config.results_dir, 'training_log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    # Load dataset
    dataset = TUDataset(root=config.data_dir, name=config.dataset_name).shuffle()

    n = len(dataset)
    train_size = int(config.train_split * n)
    val_size = int(config.val_split * n)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    labels = list(range(dataset.num_classes))

    # Analyze statistics
    dataset_stats_result = analyze_dataset_statistics(train_dataset, dataset.num_classes)
    dataset_stats = dataset_stats_result[:2]
    logger.info(f"Node Summary: {dataset_stats[0]}")
    logger.info(f"Edge Summary: {dataset_stats[1]}")
    global_max_degree = dataset_stats_result[2] 
    logger.info(f"Global Max Degree: {global_max_degree}")

    # ------------------------------------------------------------------
    # PRE-COMPUTE TARGET STATISTICS FOR GA (before model initialization)
    # ------------------------------------------------------------------
    logger.info("Computing target statistics for GA refinement...")
    target_distributions = {}
    train_graphs_by_class = {c: [] for c in labels}
    for data in train_dataset:
        train_graphs_by_class[int(data.y.item())].append(data)
        
    for c in labels:
        graphs = train_graphs_by_class[c]
        if len(graphs) > 0:
            target_distributions[c] = get_target_distribution_stats(graphs, num_bins=config.num_bins)
        else:
            target_distributions[c] = get_target_distribution_stats(train_graphs_by_class[0], num_bins=config.num_bins)

    # Pre-compute per-class average edge counts for the GA edge penalty
    class_avg_edges = {}
    for c in labels:
        class_graphs = train_graphs_by_class[c]
        if len(class_graphs) > 0:
            class_avg_edges[c] = float(
                sum(d.edge_index.size(1) for d in class_graphs) / len(class_graphs)
            ) / 2.0  # Divide by 2 for undirected logical edges
        else:
            class_avg_edges[c] = float(dataset_stats[1].get(c, {}).get('mean', 0))

    logger.info("Target statistics computed.")

    # ------------------------------------------------------------------
    # INITIALIZE MODELS
    # ------------------------------------------------------------------
    generator = Generator(config, dataset.num_classes, dataset.num_node_features).to(device)
    discriminator = Discriminator(dataset.num_node_features, dataset.num_classes, config.hidden_dim_dis).to(device)

    # ------------------------------------------------------------------
    # CONFIGURE GENERATOR GA ATTRIBUTES
    # ------------------------------------------------------------------
    # These are read by Generator.forward() when use_refinement is True.
    # use_refinement starts False and is toggled by train() after warmup.
    generator.target_distributions = target_distributions
    generator.class_avg_edges = class_avg_edges
    generator.use_refinement = False  # train() handles warmup toggle

    logger.info(f"GA config: pop={config.training_refiner_pop}, "
                f"gens={config.training_refiner_gens}, "
                f"warmup={config.refinement_warmup_epochs} epochs")

    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_gen, betas=config.betas_gen)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_dis, betas=config.betas_dis)

    # ------------------------------------------------------------------
    # TRAINING (single phase — GA runs inside generator when enabled)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info(">>> TRAINING: WGAN with In-Generator GA Refinement <<<")
    logger.info("=" * 50)
    train(generator, discriminator, train_loader, val_loader, opt_g, opt_d,
          config, labels, dataset_stats, device, logger, config.epochs)

    # Load best model
    save_path = os.path.join(config.save_dir, f'{config.dataset_name}_best_model.pt')
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        logger.info(f"Loaded best checkpoint from {save_path} (epoch {checkpoint.get('epoch', '?')})")

    # ------------------------------------------------------------------
    # FINAL EVALUATION 
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    if config.use_refinement:
        generator.use_refinement = True
        logger.info(">>> FINAL EVALUATION (Generator + GA pipeline) <<<")
    else:
        generator.use_refinement = False
        logger.info(">>> FINAL EVALUATION (Generator without GA) <<<")
    logger.info("=" * 50)

    # Collect real test graphs
    real_test_graphs = []
    for batch in test_loader:
        batch = batch.to(device)
        real_test_graphs.extend(extract_individual_graphs(batch))

    # Generate graphs using the full trained pipeline
    generated_graphs = []
    generator.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            fake_batch = generator(batch.y.size(0), dataset_stats, batch.y.to(device))
            generated_graphs.extend(extract_individual_graphs(fake_batch))

    # Save graph structures and statistics
    logger.info("Saving graph structures and statistics...")

    save_graph_structures(real_test_graphs,
                          os.path.join(config.results_dir, "real_graphs.json"), logger)
    compute_and_save_detailed_stats(real_test_graphs, config,
                                     os.path.join(config.results_dir, "real_stats.json"), logger)

    save_graph_structures(generated_graphs,
                          os.path.join(config.results_dir, "generated_graphs.json"), logger)
    compute_and_save_detailed_stats(generated_graphs, config,
                                     os.path.join(config.results_dir, "generated_stats.json"), logger)

    # Run full evaluation with file output
    logger.info("\n>>> Detailed Evaluation <<<")
    final_score = evaluate(
        generator, discriminator, labels, test_loader, train_loader,
        dataset_stats, config, device, logger, file=True
    )

    # Visualize samples
    visualize_custom_list(
        generated_graphs[:6], "Generated Samples",
        os.path.join(config.results_dir, "generated_samples.png")
    )

    # ------------------------------------------------------------------
    # SUMMARY REPORT
    # ------------------------------------------------------------------
    logger.info("Generating summary report...")
    summary_path = os.path.join(config.results_dir, "summary_report.txt")
    with open(summary_path, "w") as f:
        f.write("Experiment Summary Report\n")
        f.write("=========================\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Dataset: {config.dataset_name}\n")
        f.write(f"Seed: {config.seed}\n")
        f.write(f"GA Refinement: {'Enabled (in-training STE)' if config.use_refinement else 'Disabled'}\n\n")

        if config.use_refinement:
            f.write("GA Configuration\n")
            f.write("----------------\n")
            f.write(f"  Warmup Epochs:       {config.refinement_warmup_epochs}\n")
            f.write(f"  Training Pop Size:   {config.training_refiner_pop}\n")
            f.write(f"  Training Generations:{config.training_refiner_gens}\n")
            f.write(f"  Op Weights:          {config.refinement_op_weights}\n\n")

        f.write("Performance Metrics (MMD - Lower is Better)\n")
        f.write("-------------------------------------------\n")
        f.write(f"Final MMD Combined: {final_score:.6f}\n\n")

        f.write("Note: Final evaluation uses the same pipeline as training.\n")
        f.write("If the GA was active during training, it was also active during\n")
        f.write("evaluation. The STE bridges the GA to the edge predictor so that\n")
        f.write("discriminator gradients improve edge prediction quality.\n")

    logger.info(f"Summary report saved to: {summary_path}")
    logger.info("Experiment complete!")


# --------------------------
#  Custom Visualization Helper
# --------------------------
def visualize_custom_list(graph_list, title_prefix, save_path):
    import matplotlib.pyplot as plt
    import networkx as nx
    
    if not graph_list: return
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, g in enumerate(graph_list):
        if i >= 6: break
        ax = axes[i]
        G = nx.Graph()
        G.add_nodes_from(range(g.x.size(0)))
        if g.edge_index.size(1) > 0:
            edges = g.edge_index.cpu().numpy().T
            G.add_edges_from(edges)
        
        label = int(g.y.item())
        nx.draw(G, ax=ax, node_size=50,
                node_color='lightblue' if label == 0 else 'lightcoral')
        ax.set_title(f"{title_prefix} Cls {label}\n{G.number_of_nodes()}N, {G.number_of_edges()}E")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()