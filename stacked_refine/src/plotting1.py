import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def get_bin_edges(num_bins, max_val, bin_width):
    """
    Calculates the edges of each bin for plotting on the X-axis.
    """
    # Create bin edges: [0, width, 2*width, ...]
    edges = np.arange(0, num_bins + 1) * bin_width
    return edges

def plot_single_distribution(metric_name, class_id, data, params, title, color, output_path, y_label="Probability"):
    """
    Generates a single distribution plot (like the screenshot).
    """
    if not data:
        return

    plt.figure(figsize=(6, 4))
    
    bins_count = len(data)
    
    # Extract scaling params
    if params and metric_name in params:
        p = params[metric_name]
        bin_width = p.get('bin_width', 1.0)
        max_val = p.get('max_val', bins_count)
        # Construct edges
        edges = get_bin_edges(bins_count, max_val, bin_width)
        # Centers for bar plot
        x_axis = (edges[:-1] + edges[1:]) / 2
        width = bin_width * 0.8  # Slight gap between bars
    else:
        # Fallback
        x_axis = np.arange(bins_count)
        width = 0.8

    # Plot Bar Chart
    plt.bar(x_axis, data, width=width, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Styling
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(f"{metric_name.capitalize()}", fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    
    # Grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # For Degree, force integer ticks if feasible
    if metric_name == 'degree':
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Axis Limits
    if metric_name == 'clustering':
        plt.xlim(0, 1.0)
    elif metric_name == 'spectral':
        plt.xlim(0, 2.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate specific individual plots from stats.json files.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the timestamped results directory")
    args = parser.parse_args()

    # Define file paths
    files = {
        "Real": (os.path.join(args.dir, "real_stats.json"), "gray"),
        "Raw": (os.path.join(args.dir, "raw_stats.json"), "tab:blue"),
        "Refined": (os.path.join(args.dir, "refined_stats.json"), "tab:red")
    }

    # Load Baselines (Real) for parameters
    real_json = load_json(files["Real"][0])
    if not real_json:
        print("Error: real_stats.json is required for parameters.")
        return

    # Create output directory
    plot_dir = os.path.join(args.dir, "plots_individual")
    os.makedirs(plot_dir, exist_ok=True)

    classes = sorted([int(k) for k in real_json.keys()])
    metrics = ['degree', 'clustering', 'spectral']

    for cls_id in classes:
        cls_key = str(cls_id)
        
        # Get Real Parameters (used for all axes)
        real_cls = real_json[cls_key]
        real_params = real_cls.get("distribution_params", {})

        for label, (filepath, color) in files.items():
            data_json = load_json(filepath)
            if not data_json or cls_key not in data_json:
                continue

            dist_data = data_json[cls_key].get("distributions", {})
            
            for metric in metrics:
                if metric in dist_data:
                    output_filename = f"Class{cls_id}_{label}_{metric}.png"
                    output_path = os.path.join(plot_dir, output_filename)
                    
                    title = f"{label} - Class {cls_id} - {metric.capitalize()}"
                    
                    plot_single_distribution(
                        metric_name=metric,
                        class_id=cls_id,
                        data=dist_data[metric],
                        params=real_params, # Always use Real params for consistent X-axis
                        title=title,
                        color=color,
                        output_path=output_path
                    )

    print(f"\nIndividual plots generated in: {plot_dir}")

if __name__ == "__main__":
    main()