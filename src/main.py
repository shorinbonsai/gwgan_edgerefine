import os
import random
import logging
import argparse
import json
from datetime import datetime
from dataclasses import asdict


import numpy as np
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from config import Config
from models import Generator, Discriminator
from train import train, evaluate
from utils import visualize_graphs, analyze_dataset_statistics, get_target_distribution_stats

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
    
    # CREATE DYNAMIC DIRECTORY STRUCTURE
    # Format: results/<Dataset>/<Timestamp>_Seed<Seed>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_Seed{config.seed}"

    # Update config paths to point to this specific run directory
    config.results_dir = os.path.join(config.results_dir, config.dataset_name, run_name)
    config.save_dir = os.path.join(config.save_dir, config.dataset_name, run_name)

    # Setup directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Save the full configuration state for reproducibility
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

    # Fixed seeds for reproducibility
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
    logger.info(f"Node features: {dataset.num_node_features}, Classes: {dataset.num_classes}")

    # labels as class ids
    labels = list(range(dataset.num_classes))

    # Analyze statistics (returns node_stats, edge_stats)
    dataset_stats_result = analyze_dataset_statistics(train_dataset, dataset.num_classes)
    dataset_stats = dataset_stats_result[:2]
    logger.info(f"Node Summary: {dataset_stats[0]}")
    logger.info(f"Edge Summary: {dataset_stats[1]}")
    global_max_degree = dataset_stats_result[2] 
    logger.info(f"Global Max Degree: {global_max_degree}")

    # --- PRE-COMPUTE TARGET STATISTICS FOR RUST REFINER ---
    logger.info("Computing target statistics for Graph Refiner...")
    target_distributions = {}

    # Group training graphs by class
    train_graphs_by_class = {c: [] for c in labels}
    for data in train_dataset:
        train_graphs_by_class[int(data.y.item())].append(data)
        
    for c in labels:
        graphs = train_graphs_by_class[c]
        if len(graphs) > 0:
            # Compute histograms/stats for this class
            target_distributions[c] = get_target_distribution_stats(graphs, num_bins=config.num_bins)
        else:
            # Fallback for empty classes (shouldn't happen with standard datasets)
            logger.warning(f"Class {c} has no training samples! Using class 0 stats as fallback.")
            target_distributions[c] = get_target_distribution_stats(train_graphs_by_class[0], num_bins=config.num_bins)
    
    logger.info("Target statistics computed.")

    # Initialize models
    generator = Generator(config, dataset.num_classes, dataset.num_node_features).to(device)
    discriminator = Discriminator(dataset.num_node_features, dataset.num_classes, config.hidden_dim_dis).to(device)

    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_gen, betas=config.betas_gen)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_dis, betas=config.betas_dis)

    train(generator, discriminator, train_loader, val_loader, opt_g, opt_d, config, labels, dataset_stats, target_distributions, device, logger, config.epochs)


    # Load best model if exists
    save_path = os.path.join(config.save_dir, 'best_model.pt')
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        logger.info(f"Loaded checkpoint from {save_path}, epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        logger.warning("No checkpoint found - using current model weights")

    # Final evaluation
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    evaluate(generator, discriminator, labels, test_loader, train_loader, dataset_stats, config, device, logger, file=True)

    # Overall summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info("Lower MMD = Better distribution match (< 0.05 excellent, < 0.1 good)")
    logger.info("Higher Uniqueness = More diverse graphs (> 0.9 excellent)")
    logger.info("Higher Novelty = More creative generation (> 0.8 good)")

    # Visualize some samples
    visualize_graphs(generator, dataset_stats, 6, device, os.path.join(config.results_dir, f'{config.dataset_name}_generated_samples.png'))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
