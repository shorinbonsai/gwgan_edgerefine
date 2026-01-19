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

from config import Config
from models import Generator, Discriminator
from train import train, evaluate
from utils import visualize_graphs, analyze_dataset_statistics, get_target_distribution_stats, extract_individual_graphs, compute_graph_statistics, compute_mmd, wl_graph_hash

# Import Rust Extension
import graph_refiner

# --------------------------
# Logging
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --------------------------
#  Main Function
# --------------------------
def main():
    # 1. PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description="Graph Generation Experiment")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    args = parser.parse_args()

    config = Config()

    # Override config if CLI args are provided
    if args.seed is not None:
        config.seed = args.seed
    if args.dataset is not None:
        config.dataset_name = args.dataset
    
    # 2. CREATE DYNAMIC DIRECTORY STRUCTURE (Date/Time based)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_Seed{config.seed}"

    # Update results path to include the timestamped folder
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

    # Split dataset
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
    
    # --- PRE-COMPUTE TARGET STATISTICS FOR RUST REFINER ---
    logger.info("Computing target statistics for Graph Refiner...")
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
    logger.info("Target statistics computed.")

    # Initialize models
    generator = Generator(config, dataset.num_classes, dataset.num_node_features).to(device)
    discriminator = Discriminator(dataset.num_node_features, dataset.num_classes, config.hidden_dim_dis).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_gen, betas=config.betas_gen)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_dis, betas=config.betas_dis)

    # ------------------------------------------------------------------
    # PHASE 1: PURE WGAN TRAINING
    # ------------------------------------------------------------------
    logger.info(">>> PHASE 1: Training WGAN (No Refinement) <<<")
    train(generator, discriminator, train_loader, val_loader, opt_g, opt_d, config, labels, dataset_stats, device, logger, config.epochs)

    # Load best model
    save_path = os.path.join(config.save_dir, 'best_model.pt')
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        logger.info(f"Loaded best WGAN checkpoint from {save_path}")

    # ------------------------------------------------------------------
    # PHASE 2: POST-TRAINING REFINEMENT
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info(">>> PHASE 2: Evolutionary Edge Refinement <<<")
    logger.info("=" * 50)
    
    # Placeholders for final scores
    refined_score = 0.0
    raw_score = 0.0
    
    if config.use_refinement:
        refined_graphs_list = []
        raw_graphs_list = [] # Keep for comparison
        real_test_graphs = []
        
        # Create directory for GA logs
        ga_log_dir = os.path.join(config.results_dir, "ga_logs")
        os.makedirs(ga_log_dir, exist_ok=True)
        
        # 1. Gather all real test graphs
        for batch in test_loader:
            batch = batch.to(device)
            real_test_graphs.extend(extract_individual_graphs(batch))
            
        # 2. Generate and Refine corresponding fake graphs
        # We iterate over test_loader to ensure we generate same label distribution and count
        refiner = graph_refiner.GraphRefiner(config.refiner_pop_size)
        refiner.set_operation_weights(config.refinement_op_weights)
        refiner.set_probabilities(config.crossover_probability, config.mutation_probability)
        
        # Unpack weights/gammas
        w_tup = (config.weights['degree'], config.weights['clustering'], config.weights['spectral'])
        g_tup = (config.gammas['degree'], config.gammas['clustering'], config.gammas['spectral'])

        logger.info(f"Refining {len(real_test_graphs)} graphs...")
        
        generator.eval()
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            labels_batch = batch.y
            
            with torch.no_grad():
                # Generate raw WGAN output
                fake_batch = generator(labels_batch.size(0), dataset_stats, labels_batch)
                
            batch_raw_graphs = extract_individual_graphs(fake_batch)
            raw_graphs_list.extend(batch_raw_graphs)
            
            # Refine each graph in the batch
            for i, g in enumerate(batch_raw_graphs):
                label = int(g.y.item())
                num_nodes = g.x.size(0)
                
                if num_nodes < 3: 
                    refined_graphs_list.append(g) # Skip trivial
                    continue
                
                # Extract edges for Rust
                edges_list = []
                edge_index = g.edge_index
                for e_i in range(edge_index.size(1)):
                    u, v = int(edge_index[0, e_i].item()), int(edge_index[1, e_i].item())
                    if u < v:
                        edges_list.append((u, v))
                
                # --- Rust Refiner ---
                dynamic_gene_len = num_nodes * 2
                refiner.load_initial_graph(num_nodes, edges_list, config.seed + i + batch_idx*1000, dynamic_gene_len)
                
                stats = target_distributions[label]
                refiner.set_target_statistics(
                    stats['degree'][0], stats['degree'][1], stats['degree'][2],
                    stats['clustering'][0], stats['clustering'][1], stats['clustering'][2],
                    stats['spectral'][0], stats['spectral'][1], stats['spectral'][2],
                    w_tup, g_tup,
                    int(global_max_degree)
                )
                
                # Run Evolution
                refiner.evolve(config.refiner_gens, 42 + i)

                # 3. GA PROCESS LOGGING
                # Save logs for the first graph of each batch to monitor evolution without spamming files
                if i == 0:
                    log_name = f"batch_{batch_idx}_graph_{0}"
                    try:
                        refiner.save_logs(os.path.join(ga_log_dir, f"{log_name}_history.csv"))
                        refiner.save_results(os.path.join(ga_log_dir, f"{log_name}_best.txt"))
                    except Exception as e:
                        logger.warning(f"Failed to save GA logs: {e}")
                
                # Retrieve Best Edges
                best_edges = refiner.get_best_graph()
                
                # Reconstruct Data Object
                if len(best_edges) > 0:
                    new_edge_index = torch.tensor(best_edges, dtype=torch.long, device=device).t()
                    # Make undirected for PyG (u->v AND v->u)
                    new_edge_index = torch.cat([new_edge_index, new_edge_index.flip(0)], dim=1)
                else:
                    new_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                
                refined_g = Data(x=g.x.clone(), edge_index=new_edge_index, y=g.y.clone())
                refined_graphs_list.append(refined_g)

            if (batch_idx + 1) % 5 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

        # 3. Final Evaluation of Refined Graphs
        logger.info("\n>>> Evaluating Refined Graphs <<<")
        refined_save_path = os.path.join(config.results_dir, f"{config.dataset_name}_refined_results.txt") 
        refined_score = evaluate_graph_sets(refined_graphs_list, real_test_graphs, train_loader, labels, config, logger, phase="REFINED", save_path=refined_save_path) # NEW: Passed train_loader and save_path

        logger.info("\n>>> Evaluating Raw WGAN Graphs (Pre-Refinement) <<<")
        raw_save_path = os.path.join(config.results_dir, f"{config.dataset_name}_raw_results.txt") # NEW
        raw_score = evaluate_graph_sets(raw_graphs_list, real_test_graphs, train_loader, labels, config, logger, phase="RAW", save_path=raw_save_path) # NEW: Passed train_loader and save_path

        # Visualize
        visualize_custom_list(refined_graphs_list[:6], "Refined Samples", os.path.join(config.results_dir, "refined_samples.png"))
        visualize_custom_list(raw_graphs_list[:6], "Raw Samples", os.path.join(config.results_dir, "raw_samples.png"))


        # 4. SUMMARY REPORT
        logger.info("Generating summary report...")
        summary_path = os.path.join(config.results_dir, "summary_report.txt")
        with open(summary_path, "w") as f:
            f.write("Experiment Summary Report\n")
            f.write("=========================\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Dataset: {config.dataset_name}\n")
            f.write(f"Seed: {config.seed}\n\n")
            
            f.write("Performance Metrics (MMD - Lower is Better)\n")
            f.write("-------------------------------------------\n")

            f.write(f"Raw WGAN MMD:      {raw_score:.6f}\n")
            f.write(f"Refined Graph MMD: {refined_score:.6f}\n")
            improvement = raw_score - refined_score
            f.write(f"Net Improvement:   {improvement:.6f}\n")
            if improvement > 0:
                f.write("Result: Refinement IMPROVED the distribution matching.\n")
            else:
                f.write("Result: Refinement DID NOT improve the distribution matching.\n")
            f.write("\nConfiguration Details\n")
            f.write(f"Refiner Generations: {config.refiner_gens}\n")
            f.write(f"Refiner Pop Size:    {config.refiner_pop_size}\n")
        
        logger.info(f"Summary report saved to: {summary_path}")
    else:
        logger.info("Refinement skipped (config.use_refinement = False)")

    logger.info("Experiment complete!")

# --------------------------
#  Custom Helpers for List-based Eval
# --------------------------
def evaluate_graph_sets(fake_graphs, real_graphs, train_loader, labels, config, logger, phase="TEST", save_path=None):
    # Group by class
    real_by_class = {c: [] for c in labels}
    fake_by_class = {c: [] for c in labels}

    for g in real_graphs: 
        if g.y is not None: real_by_class[int(g.y.item())].append(g)
    for g in fake_graphs: 
        if g.y is not None: fake_by_class[int(g.y.item())].append(g)


    # Compute Overall MMD
    logger.info(f"\n--- {phase} EVALUATION RESULTS ---")
    
    real_stats_all = compute_graph_statistics(real_graphs, num_bins=config.num_bins)
    fake_stats_all = compute_graph_statistics(fake_graphs, num_bins=config.num_bins)

    mmd_deg = compute_mmd(real_stats_all['degrees'], fake_stats_all['degrees'], kernel='rbf', gamma=config.gammas['degree'])
    mmd_clus = compute_mmd(real_stats_all['clustering'], fake_stats_all['clustering'], kernel='rbf', gamma=config.gammas['clustering'])
    mmd_spec = compute_mmd(real_stats_all['spectral'], fake_stats_all['spectral'], kernel='rbf', gamma=config.gammas['spectral'])
    
    combined = (config.weights['degree']*mmd_deg + config.weights['clustering']*mmd_clus + config.weights['spectral']*mmd_spec)
    
    logger.info(f"Overall MMD: {combined:.6f} (Deg: {mmd_deg:.4f}, Clus: {mmd_clus:.4f}, Spec: {mmd_spec:.4f})")
    
    output_lines = []
    output_lines.append(f"{phase} Graph Generation Evaluation Results")
    output_lines.append("=" * 50 + "\n")
    output_lines.append(f"  Overall MMD Degree: {mmd_deg:.6f}")
    output_lines.append(f"  Overall MMD Clustering: {mmd_clus:.6f}")
    output_lines.append(f"  Overall MMD Spectral: {mmd_spec:.6f}")
    output_lines.append(f"  Overall MMD Combined: {combined:.6f}")

    # Pre-compute training hashes for novelty calculation (per class)
    train_hashes_by_class = {c: set() for c in labels}
    device = fake_graphs[0].x.device if len(fake_graphs) > 0 else torch.device('cpu')
    
    if train_loader:
        for batch in train_loader:
            if hasattr(batch, 'batch'):
                batch = batch.to(device)
                b_graphs = extract_individual_graphs(batch)
                for bg in b_graphs:
                    if bg.y is not None:
                         train_hashes_by_class[int(bg.y.item())].add(wl_graph_hash(bg))

    # Per Class Stats
    for c in labels:
        if len(real_by_class[c]) == 0 or len(fake_by_class[c]) == 0: continue
        
        r_graphs = real_by_class[c]
        f_graphs = fake_by_class[c]
        
        r_stats = compute_graph_statistics(r_graphs, config.num_bins)
        f_stats = compute_graph_statistics(f_graphs, config.num_bins)
        
        # Per Class MMDs
        m_deg = compute_mmd(r_stats['degrees'], f_stats['degrees'], gamma=config.gammas['degree'])
        m_clus = compute_mmd(r_stats['clustering'], f_stats['clustering'], gamma=config.gammas['clustering'])
        m_spec = compute_mmd(r_stats['spectral'], f_stats['spectral'], gamma=config.gammas['spectral'])
        m_comb = (config.weights['degree']*m_deg + config.weights['clustering']*m_clus + config.weights['spectral']*m_spec)

        # Calculate Uniqueness/Novelty
        fake_hashes = [wl_graph_hash(g) for g in f_graphs]
        uniqueness = len(set(fake_hashes)) / len(fake_hashes) if fake_hashes else 0.0
        
        # Novelty: check against training hashes of same class
        t_hashes = train_hashes_by_class[c]
        novel_count = sum(1 for h in fake_hashes if h not in t_hashes)
        novelty = novel_count / len(fake_hashes) if fake_hashes else 0.0
        
        logger.info(f"Class {c}: Nodes {np.mean(f_stats['num_nodes']):.1f} | MMD {m_comb:.4f} | Uni {uniqueness:.2f} | Nov {novelty:.2f}")

        output_lines.append(f"Class {c}:")
        output_lines.append(f"  Sample size: {len(r_graphs)} real, {len(f_graphs)} generated")
        output_lines.append(f"  MMD Degree: {m_deg:.6f}")
        output_lines.append(f"  MMD Clustering: {m_clus:.6f}")
        output_lines.append(f"  MMD Spectral: {m_spec:.6f}")
        output_lines.append(f"  MMD Combined: {m_comb:.6f}")
        output_lines.append(f"  Uniqueness: {uniqueness:.6f}")
        output_lines.append(f"  Novelty: {novelty:.6f}")
        output_lines.append(f"  Avg Nodes: {np.mean(r_stats['num_nodes']):.1f} -> {np.mean(f_stats['num_nodes']):.1f}")
        output_lines.append(f"  Avg Edges: {np.mean(r_stats['num_edges']):.1f} -> {np.mean(f_stats['num_edges']):.1f}\n")

    # Write to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(output_lines))
        logger.info(f"Detailed results saved to: {save_path}")

    return combined

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
        nx.draw(G, ax=ax, node_size=50, node_color='lightblue' if label==0 else 'lightcoral')
        ax.set_title(f"{title_prefix} Cls {label}\n{G.number_of_nodes()}N, {G.number_of_edges()}E")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    main()