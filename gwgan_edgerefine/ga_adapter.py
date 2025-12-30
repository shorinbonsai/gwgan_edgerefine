import torch
import numpy as np
import logging
import sys

# Attempt to import the Rust extension
try:
    import gwgan_edgerefine_rs
except ImportError as e:
    # We print a warning but don't crash, allowing the GAN to run without GA if needed.
    print(f"\n[WARNING] Could not import Rust GA library: {e}")
    print("Evolutionary refinement will be DISABLED.")
    print("Ensure you have run 'python build_local.py' to compile the extension.\n")
    gwgan_edgerefine_rs = None

class GARefiner:
    def __init__(self, config, dataset_stats):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.enabled = gwgan_edgerefine_rs is not None

    def refine_batch(self, generated_graphs_batch, real_graphs_batch):
        """
        Refine a batch of PyG Data objects using the Rust GA.
        """
        if not self.enabled:
            return generated_graphs_batch

        # 1. Compute Target Distributions (Average of real batch)
        # Assuming utils.compute_graph_stats is available and imported correctly
        # Depending on where main.py runs, imports might need adjustment. 
        # We attempt a relative import fallback.
        try:
            from gwgan_edgerefine.utils import compute_graph_stats
        except ImportError:
            try:
                from utils import compute_graph_stats
            except ImportError:
                self.logger.error("Could not import 'utils.compute_graph_stats'. Skipping refinement.")
                return generated_graphs_batch
        
        # This returns a dictionary with 'degrees' and 'clustering' matrices (N_graphs x num_bins)
        real_stats = compute_graph_stats(real_graphs_batch, num_bins=self.config.num_bins)
        
        # Average the histograms to get a single target distribution for this batch
        target_deg_hist = np.mean(real_stats['degrees'], axis=0).astype(np.float32)
        target_clust_hist = np.mean(real_stats['clustering'], axis=0).astype(np.float32)

        refined_graphs = []

        for graph in generated_graphs_batch:
            # 2. Convert PyG Graph to Adjacency Matrix (numpy)
            num_nodes = graph.num_nodes
            adj_mat = np.zeros((num_nodes, num_nodes), dtype=np.uint)
            
            # Ensure edges are undirected for the matrix
            edges = graph.edge_index.cpu().numpy()
            if edges.shape[1] > 0:
                adj_mat[edges[0], edges[1]] = 1
                adj_mat[edges[1], edges[0]] = 1 # Symmetry

            # 3. Call Rust GA
            try:
                # Parameters from config or defaults
                gens = getattr(self.config, 'ga_generations', 10)
                pop_size = getattr(self.config, 'ga_pop_size', 20)
                
                # The Rust function returns a refined Adjacency Matrix
                refined_adj = gwgan_edgerefine_rs.refine_graph(
                    adj_mat.astype(np.uint), 
                    target_deg_hist,
                    target_clust_hist,
                    gens,
                    pop_size,
                    self.config.num_bins
                )
                
                # 4. Convert back to PyG Data
                # Get indices where value is 1
                rows, cols = np.where(refined_adj > 0)
                edge_index = torch.tensor([rows, cols], dtype=torch.long)
                
                # Create new Data object preserving attributes
                new_data = graph.clone()
                new_data.edge_index = edge_index
                refined_graphs.append(new_data)

            except Exception as e:
                self.logger.error(f"Error during Rust GA refinement: {e}")
                refined_graphs.append(graph) # Fallback to original

        return refined_graphs