import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize all graphs from JSON results.")
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="./results", 
        help="Path to the directory containing the graph JSON files. If not provided, searches current directory."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./all_visualizations2", 
        help="Directory where images will be saved."
    )
    return parser.parse_args()

def load_json_file(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_dataset_name_from_path(filepath):
    try:
        abs_path = os.path.abspath(filepath)
        parts = abs_path.split(os.sep)
        if 'results' in parts:
            idx = len(parts) - 1 - parts[::-1].index('results')
            if idx + 1 < len(parts):
                dataset_candidate = parts[idx + 1]
                if not dataset_candidate.startswith('202'):
                    return dataset_candidate
        directory = os.path.dirname(abs_path)
        grandparent = os.path.dirname(directory)
        return os.path.basename(grandparent)
    except Exception:
        return "Unknown_Dataset"

def convert_to_networkx(graph_data):
    """
    Robustly converts graph data to NetworkX.
    Prioritizes the specific {"edges": [[u,v]...], "num_nodes": N} format found in the repo.
    """
    G = nx.Graph()

    # --- PRIORITY 1: Handle User's Specific Dictionary Format ---
    if isinstance(graph_data, dict):
        # Format: {"edges": [[0,1], [0,2]...], "num_nodes": 10, ...}
        if 'edges' in graph_data and isinstance(graph_data['edges'], (list, tuple)):
            # 1. Add nodes (ensures isolated nodes are included)
            if 'num_nodes' in graph_data:
                G.add_nodes_from(range(int(graph_data['num_nodes'])))
            
            # 2. Add edges from the edge list
            for edge in graph_data['edges']:
                # Ensure edge is a valid pair [u, v]
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    G.add_edge(int(edge[0]), int(edge[1]))
            
            return G

        # Check for PyTorch Geometric style 'edge_index'
        if 'edge_index' in graph_data:
            edges = graph_data['edge_index']
            if isinstance(edges, (list, tuple)) and len(edges) == 2 and isinstance(edges[0], (list, tuple)):
                row, col = edges[0], edges[1]
                if 'num_nodes' in graph_data:
                    G.add_nodes_from(range(graph_data['num_nodes']))
                
                for u, v in zip(row, col):
                    G.add_edge(int(u), int(v))
                return G
                
        # Check for common nested wrappers like 'adj' or 'graph'
        potential_data_keys = ['adj', 'adjacency', 'adjacency_list', 'adj_list', 'graph', 'data']
        for key in potential_data_keys:
            if key in graph_data:
                return convert_to_networkx(graph_data[key])

    # --- PRIORITY 2: List Formats ---
    if isinstance(graph_data, list):
        num_nodes = len(graph_data)
        G.add_nodes_from(range(num_nodes))
        
        for i, row in enumerate(graph_data):
            if not isinstance(row, (list, tuple)):
                continue

            # Matrix Detection: Row length equals num_nodes AND all 0/1
            is_matrix = (len(row) == num_nodes) and all(isinstance(x, (int, float)) and int(x) in [0, 1] for x in row)
            
            if is_matrix:
                for j, val in enumerate(row):
                    if int(val) == 1 and i < j: # Upper triangle
                        G.add_edge(i, j)
            else:
                # Adjacency List: [neighbor_idx, neighbor_idx, ...]
                for neighbor in row:
                    if isinstance(neighbor, (int, float)):
                        n_idx = int(neighbor)
                        if i < n_idx: # Undirected assumption
                            G.add_edge(i, n_idx)

    # --- PRIORITY 3: Fallback Dictionary Formats ---
    elif isinstance(graph_data, dict):
        # NetworkX Node-Link
        if 'nodes' in graph_data and 'links' in graph_data:
            try:
                return nx.node_link_graph(graph_data)
            except Exception:
                pass
        
        # Adjacency Dictionary {"0": [1, 2], ...}
        # Only proceed if keys look like integers (to avoid the "5 nodes" bug)
        is_likely_adj_dict = True
        try:
            sample_keys = list(graph_data.keys())[:5]
            for k in sample_keys:
                int(k) 
        except ValueError:
            is_likely_adj_dict = False

        if is_likely_adj_dict:
            for u_str, neighbors in graph_data.items():
                try:
                    u = int(u_str)
                except ValueError:
                    u = u_str
                
                G.add_node(u)
                if isinstance(neighbors, (list, tuple)):
                    for v in neighbors:
                        if isinstance(v, (str, int, float)):
                            try:
                                v_int = int(v)
                                if u != v_int: 
                                    G.add_edge(u, v_int)
                            except ValueError:
                                G.add_edge(u, v)

    return G

def visualize_and_save(graphs, category, output_base_dir, dataset_name):
    if not graphs:
        return

    save_dir = os.path.join(output_base_dir, dataset_name, category)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating {len(graphs)} visualizations for '{category}' in {save_dir}...")

    # Define color palette for classes
    # 0: Blue, 1: Red, 2: Green, 3: Purple, 4: Orange, 5: Yellow, etc.
    CLASS_COLORS = [
        '#3498db', # Class 0: Blue (Default)
        '#e74c3c', # Class 1: Red
        '#2ecc71', # Class 2: Green
        '#9b59b6', # Class 3: Purple
        '#e67e22', # Class 4: Orange
        '#f1c40f', # Class 5: Yellow
        '#34495e', # Class 6: Navy
        '#1abc9c', # Class 7: Teal
    ]

    for graph_id, graph_data in enumerate(graphs):
        G = convert_to_networkx(graph_data)
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Determine Node Color based on Label
        node_color = CLASS_COLORS[0] # Default to Class 0 color
        label_str = ""
        
        if isinstance(graph_data, dict) and 'label' in graph_data:
            try:
                raw_label = graph_data['label']
                # Handle cases where label might be a single-item list like [1]
                if isinstance(raw_label, list) and len(raw_label) > 0:
                    label_val = int(raw_label[0])
                else:
                    label_val = int(raw_label)
                
                label_str = f" | Class: {label_val}"
                
                # Select color, using modulo to wrap around if classes exceed palette size
                color_idx = label_val % len(CLASS_COLORS)
                node_color = CLASS_COLORS[color_idx]
            except (ValueError, TypeError):
                # Fallback if label is not convertible to int
                label_str = f" | Label: {graph_data['label']}"
        
        # Simple Spring Layout
        plt.figure(figsize=(8, 8))
        try:
            pos = nx.spring_layout(G, seed=42) 
            nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_color, alpha=0.9)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
            
            plt.title(f"ID: {graph_id}{label_str} | Nodes: {num_nodes} | Edges: {num_edges}", fontsize=12, fontweight='bold')
            plt.axis('off')
            
            filename = f"{category}_graph_{graph_id}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"Error plotting graph {graph_id}: {e}")
        finally:
            plt.close()

    print(f"Finished {category}.")

def main():
    args = parse_args()
    search_path = args.results_dir
    files_to_process = {
        'real': os.path.join(search_path, 'real_graphs.json'),
        'raw': os.path.join(search_path, 'raw_graphs.json'),
        'refined': os.path.join(search_path, 'refined_graphs.json')
    }

    print(f"Looking for graph files in: {os.path.abspath(search_path)}")
    found_any = False
    
    # 1. Check direct paths
    for category, filepath in files_to_process.items():
        if os.path.exists(filepath):
            dataset_name = get_dataset_name_from_path(filepath)
            print(f"Found {category} graphs: {filepath}")
            graphs = load_json_file(filepath)
            visualize_and_save(graphs, category, args.output_dir, dataset_name)
            found_any = True
    
    # 2. Recursive search
    if not found_any:
        print("Direct files not found. Searching recursively...")
        for category in ['real', 'raw', 'refined']:
            found_files = glob.glob(os.path.join(search_path, "**", f"{category}_graphs.json"), recursive=True)
            for filepath in found_files:
                dataset_name = get_dataset_name_from_path(filepath)
                print(f"Found {category} graphs: {filepath}")
                graphs = load_json_file(filepath)
                visualize_and_save(graphs, category, args.output_dir, dataset_name)
                found_any = True
    
    if not found_any:
        print("\nERROR: No graph files found.")

if __name__ == "__main__":
    main()