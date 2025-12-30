import torch
import numpy as np
import logging

# With maturin, the rust library is installed as a standard python package.
# We no longer need to manually append sys.path.
try:
    import gwgan_edgerefine_rs
except ImportError:
    gwgan_edgerefine_rs = None

class GARefiner:
    def __init__(self, config, dataset_stats):
        """
        config: Configuration object containing GA params.
        dataset_stats: Tuple (node_stats, edge_stats) or similar.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.enabled = gwgan_edgerefine_rs is not None
        if not self.enabled:
            self.logger.warning(
                "Rust GA library (gwgan_edgerefine_rs) not found. "
                "Evolutionary refinement will be skipped. "
                "Ensure you have installed the package via 'maturin develop' or 'pip install .'."
            )

    def refine_batch(self, generated_graphs_batch, real_graphs_batch):
        """
        Refine a batch of PyG Data objects using the Rust GA.
        
        Args:
            generated_graphs_batch (List[Data]): List of PyG Data objects (generated).
            real_graphs_batch (List[Data]): List of PyG Data objects (real) to compute targets.
        
        Returns:
            List[Data]: The refined graphs.
        """
        if not self.enabled:
            return generated_graphs_batch

        # 1. Compute Target Distributions (Average of real batch)
        # We need the histograms for Degree and Clustering.
        # We assume utils.compute_graph_stats is available.
        from utils import compute_graph_stats
        
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
                # Note: We cast to uint because Rust expects usize (which matches u64/uint on 64-bit systems)
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