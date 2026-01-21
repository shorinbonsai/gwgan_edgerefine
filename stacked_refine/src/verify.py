import os
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

# Import necessary utils from your project
from config import Config
from utils import compute_graph_statistics

# --------------------------
# Configuration
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
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

def get_bin_edges(num_bins, max_val, bin_width):
    return np.arange(0, num_bins + 1) * bin_width

def plot_distribution(metric_name, class_id, data, bin_width, max_val, output_path):
    plt.figure(figsize=(6, 4))
    
    # Construct edges and centers
    num_bins = len(data)
    edges = get_bin_edges(num_bins, max_val, bin_width)
    x_axis = (edges[:-1] + edges[1:]) / 2
    width = bin_width * 0.8

    # Plot
    plt.bar(x_axis, data, width=width, color='gray', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Styling
    plt.title(f"PROTEINS Ground Truth - Class {class_id} - {metric_name.capitalize()}", fontsize=12)
    plt.xlabel(f"{metric_name.capitalize()}", fontsize=10)
    plt.ylabel("Probability", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    if metric_name == 'degree':
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    elif metric_name == 'clustering':
        plt.xlim(0, 1.0)
    elif metric_name == 'spectral':
        plt.xlim(0, 2.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Verify PROTEINS dataset statistics directly.")
    parser.add_argument("--dataset", type=str, default="PROTEINS", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="dataset_verification", help="Output directory")
    args = parser.parse_args()

    config = Config() # Load default config for bin settings (default 10)
    
    # 1. Load Dataset Directly
    logger.info(f"Loading {args.dataset}...")
    dataset = TUDataset(root=config.data_dir, name=args.dataset)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Group by Class
    graphs_by_class = {}
    for data in dataset:
        label = int(data.y.item())
        if label not in graphs_by_class:
            graphs_by_class[label] = []
        graphs_by_class[label].append(data)

    stats_output = {}

    # 3. Compute Stats Per Class
    for label, graphs in graphs_by_class.items():
        logger.info(f"Processing Class {label} ({len(graphs)} graphs)...")
        
        # Determine Max Degree for this class to set bin width correctly
        current_max_deg = 0
        for g in graphs:
            if g.edge_index.numel() > 0:
                d = degree(g.edge_index[0], g.x.size(0))
                m = d.max().item()
                if m > current_max_deg:
                    current_max_deg = m
        
        # Calculate Params
        # Note: If max_deg is small (e.g. 5) and bins is 10, width is 0.5
        deg_bin_width = current_max_deg / config.num_bins if current_max_deg > 0 else 1.0
        clus_bin_width = 1.0 / config.num_bins
        spec_bin_width = 2.0 / config.num_bins

        # Compute Histograms
        raw_stats = compute_graph_statistics(graphs, num_bins=config.num_bins)
        
        avg_deg = np.mean(raw_stats['degrees'], axis=0).tolist()
        avg_clus = np.mean(raw_stats['clustering'], axis=0).tolist()
        avg_spec = np.mean(raw_stats['spectral'], axis=0).tolist()

        # Save Plots
        plot_distribution('degree', label, avg_deg, deg_bin_width, current_max_deg, 
                          os.path.join(args.output_dir, f"Class{label}_Truth_Degree.png"))
        
        plot_distribution('clustering', label, avg_clus, clus_bin_width, 1.0, 
                          os.path.join(args.output_dir, f"Class{label}_Truth_Clustering.png"))
        
        plot_distribution('spectral', label, avg_spec, spec_bin_width, 2.0, 
                          os.path.join(args.output_dir, f"Class{label}_Truth_Spectral.png"))

        stats_output[label] = {
            "degree": avg_deg,
            "clustering": avg_clus,
            "spectral": avg_spec,
            "params": {
                "degree_max": int(current_max_deg),
                "degree_width": deg_bin_width
            }
        }

    # Save JSON for reference
    json_path = os.path.join(args.output_dir, "verified_stats.json")
    with open(json_path, 'w') as f:
        json.dump(stats_output, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Verified statistics saved to {json_path}")

if __name__ == "__main__":
    main()