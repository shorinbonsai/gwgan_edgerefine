import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Optional

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

# Import Rust Extension
import graph_refiner


# --------------------------
# Edge Predictor
# --------------------------
class EdgePredictor(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_edge_types: int = 1):
        super().__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 4), num_edge_types)
        )
        self.num_edge_types = num_edge_types

    def forward(self, node_features: torch.Tensor, temperature: float = 1.0):
        device = node_features.device
        n = node_features.size(0)
        h = F.relu(self.node_transform(node_features))

        if n < 2:
            return (
                torch.empty((0, self.num_edge_types), device=device),
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.zeros((n, n), device=device)
            )

        idx_i, idx_j = [], []
        for i in range(n - 1):
            idx_i.append(torch.full((n - i - 1,), i, dtype=torch.long, device=device))
            idx_j.append(torch.arange(i + 1, n, dtype=torch.long, device=device))
        idx_i = torch.cat(idx_i)
        idx_j = torch.cat(idx_j)

        pair_feats = torch.cat([h[idx_i], h[idx_j]], dim=1)
        logits = self.edge_mlp(pair_feats)
        probs = torch.sigmoid(logits / max(1e-6, temperature))
        pair_index = torch.stack([idx_i, idx_j], dim=0)

        # Build soft_adj — identical scatter logic to DistanceEdgePredictor
        soft_adj = torch.zeros((n, n), device=device)
        soft_adj[pair_index[0], pair_index[1]] = probs.squeeze(1)
        soft_adj[pair_index[1], pair_index[0]] = probs.squeeze(1)

        return probs, pair_index, soft_adj


# --------------------------
# Distance Edge Predictor (Updated with Soft Adjacency)
# --------------------------
class DistanceEdgePredictor(nn.Module):
    """Predicts edges based on learned distance in latent space.

    Returns a soft adjacency matrix alongside the pair-based format.
    The soft adjacency matrix retains gradients back to node_encoder and
    threshold, enabling the STE gradient path from the discriminator
    through edge weights.
    """
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU()
        )
        # Learnable threshold for edge creation
        self.threshold = nn.Parameter(torch.tensor(1.0))

    def forward(self, node_features: torch.Tensor, temperature: float = 1.0):
        """
        Returns:
            probs:      (M, 1) edge probabilities for each upper-triangle pair
            pair_index: (2, M) [i, j] indices for each pair
            soft_adj:   (N, N) symmetric soft adjacency matrix, retains gradients
        """
        device = node_features.device
        n = node_features.size(0)

        if n < 2:
            return (
                torch.empty((0, 1), device=device),
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.zeros((n, n), device=device)
            )

        # Encode nodes
        h = self.node_encoder(node_features)

        # Compute pairwise distances (loop over upper triangle)
        idx_i, idx_j = [], []
        distances = []

        for i in range(n - 1):
            for j in range(i + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                dist = torch.norm(h[i] - h[j], p=2)
                distances.append(dist)

        distances = torch.stack(distances).unsqueeze(1)

        # Convert distances to probabilities (closer = higher probability)
        probs = torch.sigmoid((-distances + self.threshold) / max(temperature, 1e-6))

        pair_index = torch.stack([
            torch.tensor(idx_i, device=device),
            torch.tensor(idx_j, device=device)
        ], dim=0)

        # Build soft_adj by scattering the computed probs into an N×N matrix.
        # The diagonal stays zero from initialization (loop never computes i==j).
        soft_adj = torch.zeros((n, n), device=device)
        soft_adj[pair_index[0], pair_index[1]] = probs.squeeze(1)
        soft_adj[pair_index[1], pair_index[0]] = probs.squeeze(1)  # symmetric

        return probs, pair_index, soft_adj


# --------------------------
# Generator (Updated with In-Training GA Refinement + STE)
# --------------------------
class Generator(nn.Module):
    def __init__(self, config, num_classes: int, node_feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = config.noise_dim
        self.class_embedding = nn.Embedding(num_classes, config.class_embed_dim)
        self.node_generator = nn.Sequential(
            nn.Linear(config.noise_dim + config.class_embed_dim, config.hidden_dim_gen),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_gen, node_feat_dim)
        )
        self.edge_predictor = DistanceEdgePredictor(node_feat_dim, config.hidden_dim_gen)

        # ------------------------------------------------------------------
        # GA Refinement Attributes
        # Populated by the main script before training begins.
        # When use_refinement is False the generator behaves like the original:
        # top-k edges become the output with edge_weight from soft_adj.
        # ------------------------------------------------------------------
        self.use_refinement: bool = False
        self.target_distributions: Optional[Dict] = None
        self.class_avg_edges: Optional[Dict[int, float]] = None

        # GA hyperparameters (training-time defaults — smaller than post-hoc)
        self.refiner_pop_size: int = getattr(config, 'training_refiner_pop', 50)
        self.refiner_gens: int = getattr(config, 'training_refiner_gens', 20)
        self.refiner_seed: int = getattr(config, 'seed', 42)
        self.refinement_op_weights: list = getattr(
            config, 'refinement_op_weights', [1.0] * 9
        )
        self.crossover_probability: float = getattr(config, 'crossover_probability', 0.5)
        self.mutation_probability: float = getattr(config, 'mutation_probability', 0.8)
        self.ga_weights: tuple = (
            config.weights.get('degree', 0.3),
            config.weights.get('clustering', 0.4),
            config.weights.get('spectral', 0.3)
        )
        self.ga_gammas: tuple = (
            config.gammas.get('degree', 0.01),
            config.gammas.get('clustering', 0.01),
            config.gammas.get('spectral', 0.1)
        )
        self.edge_penalty_weight: float = getattr(config, 'edge_penalty_weight', 0.01)

        # Counter for deterministic but non-repeating GA seeds
        self._call_counter: int = 0

    def sample_node_count(self, class_label: int, node_stats: Dict):
        if node_stats is None or class_label not in node_stats:
            return int(np.random.randint(10, 30))
        stats = node_stats[class_label]
        mean, std = stats['mean'], max(stats['std'], 1.0)
        count = int(np.round(np.random.normal(mean, std * 0.5)))
        return int(np.clip(count, stats['min'], stats['max']))

    # ------------------------------------------------------------------
    # GA helpers
    # ------------------------------------------------------------------
    def _refine_single_graph(
        self,
        edges_list: list,
        num_nodes: int,
        label: int,
        graph_seed: int,
    ) -> list:
        """Run Rust GA on one graph. Returns list of (u,v) refined edges."""
        refiner = graph_refiner.GraphRefiner(self.refiner_pop_size)
        refiner.set_operation_weights(self.refinement_op_weights)
        refiner.set_probabilities(self.crossover_probability, self.mutation_probability)

        dynamic_gene_len = num_nodes * 2
        refiner.load_initial_graph(num_nodes, edges_list, graph_seed, dynamic_gene_len)

        stats = self.target_distributions[label]
        avg_edges = (self.class_avg_edges.get(label, 0.0)
                     if self.class_avg_edges else 0.0)

        refiner.set_target_statistics(
            stats['degree'][0], stats['degree'][1], stats['degree'][2],
            stats['clustering'][0], stats['clustering'][1], stats['clustering'][2],
            stats['spectral'][0], stats['spectral'][1], stats['spectral'][2],
            self.ga_weights,
            self.ga_gammas,
            avg_edges,
            self.edge_penalty_weight
        )

        refiner.evolve(self.refiner_gens, graph_seed)
        return refiner.get_best_graph()

    def _build_edge_weight_via_ste(
        self,
        refined_edges: list,
        soft_adj: torch.Tensor,
        node_offset: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        STE: edge_index from GA, edge_weight from soft_adj lookup.

        For each (u,v) the GA kept or added, soft_adj[u,v] provides a
        differentiable weight that carries gradients back to the edge predictor.
        """
        if len(refined_edges) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, device=device)
            )

        us, vs, weights = [], [], []
        for (u, v) in refined_edges:
            weights.append(soft_adj[u, v])
            us.append(u + node_offset)
            vs.append(v + node_offset)

        w = torch.stack(weights)
        src = torch.tensor(us, dtype=torch.long, device=device)
        dst = torch.tensor(vs, dtype=torch.long, device=device)

        edge_index = torch.cat([
            torch.stack([src, dst], dim=0),
            torch.stack([dst, src], dim=0)
        ], dim=1)
        edge_weight = torch.cat([w, w])

        return edge_index, edge_weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        num_graphs: int,
        dataset_stats: Tuple[Dict, Dict],
        class_labels: torch.Tensor = None,
        temperature: float = 1.0
    ) -> Data:
        device = next(self.parameters()).device
        if class_labels is None:
            class_labels = torch.randint(
                0, self.num_classes, (num_graphs,), device=device
            )
        else:
            class_labels = class_labels.to(device)

        class_embeds = self.class_embedding(class_labels)
        node_stats, edge_stats = dataset_stats

        all_x, all_edges, all_weights, all_batch = [], [], [], []
        node_offset = 0
        self._call_counter += 1

        for i in range(num_graphs):
            label = int(class_labels[i].item())
            num_nodes = self.sample_node_count(label, node_stats)
            z = torch.randn(num_nodes, self.noise_dim, device=device)
            class_expand = class_embeds[i].unsqueeze(0).repeat(num_nodes, 1)
            x = self.node_generator(torch.cat([z, class_expand], dim=1))

            # Edge predictor — returns soft_adj as 3rd output
            edge_probs, pair_idx, soft_adj = self.edge_predictor(x, temperature)
            m_pairs = pair_idx.size(1) if pair_idx.numel() > 0 else 0

            # Density from dataset statistics
            density = 0.05
            if edge_stats and node_stats and label in edge_stats:
                mean_edges = max(0.0, float(edge_stats[label]['mean']))
                mean_nodes = max(1.0, float(node_stats[label]['mean']))
                density = float(np.clip(
                    mean_edges / max((mean_nodes * (mean_nodes - 1) / 2), 1),
                    0.0, 1.0
                ))

            # --- PATH A: GA refinement ---
            if (self.use_refinement
                    and self.target_distributions is not None
                    and label in self.target_distributions
                    and num_nodes >= 3
                    and m_pairs > 0):

                topk = int(round(density * m_pairs))
                initial_edges = []
                if topk > 0:
                    _, idx_topk = torch.topk(
                        edge_probs.view(-1), k=topk, largest=True
                    )
                    sel = pair_idx[:, idx_topk]
                    for k in range(sel.size(1)):
                        a, b = int(sel[0, k].item()), int(sel[1, k].item())
                        initial_edges.append((min(a, b), max(a, b)))

                graph_seed = (
                    self.refiner_seed
                    + self._call_counter * 100000
                    + i
                )
                refined_edges = self._refine_single_graph(
                    initial_edges, num_nodes, label, graph_seed
                )
                edges_tensor, weights_tensor = self._build_edge_weight_via_ste(
                    refined_edges, soft_adj, node_offset, device
                )

            # --- PATH B: Top-k only (no GA) ---
            else:
                edges_tensor = torch.empty((2, 0), dtype=torch.long, device=device)
                weights_tensor = torch.empty(0, device=device)
                topk = int(round(density * m_pairs))
                if m_pairs > 0 and topk > 0:
                    _, idx_topk = torch.topk(edge_probs.view(-1), k=topk, largest=True)
                    sel_pairs = pair_idx[:, idx_topk]
                    u = sel_pairs[0] + node_offset
                    v = sel_pairs[1] + node_offset
                    weights_tensor = torch.cat([soft_adj[sel_pairs[0], sel_pairs[1]]] * 2)
                    edges_tensor = torch.cat([
                        torch.stack([u, v], dim=0),
                        torch.stack([v, u], dim=0)
                    ], dim=1)

            all_x.append(x)
            all_edges.append(edges_tensor)
            all_weights.append(weights_tensor)
            all_batch.append(
                torch.full((num_nodes,), i, dtype=torch.long, device=device)
            )
            node_offset += num_nodes

        # Assemble batched output
        out_feat_dim = self.node_generator[-1].out_features
        x = (torch.cat(all_x, dim=0) if all_x
             else torch.empty((0, out_feat_dim), device=device))
        edge_index = (torch.cat(all_edges, dim=1) if all_edges
                      else torch.empty((2, 0), dtype=torch.long, device=device))
        edge_weight = (torch.cat(all_weights, dim=0) if all_weights
                       else torch.empty(0, device=device))
        batch = (torch.cat(all_batch, dim=0) if all_batch
                 else torch.empty((0,), dtype=torch.long, device=device))
        y = class_labels.clone().to(device)

        data = Data(x=x, edge_index=edge_index, batch=batch, y=y)
        data.edge_weight = edge_weight
        return data


# --------------------------
# Discriminator (Updated with Edge Weight Support)
# --------------------------
class Discriminator(nn.Module):
    """
    GCN-based discriminator with edge_weight support.

    When edge_weight is provided (from the generator's STE output),
    GCNConv scales neighbor contributions during message passing.
    For real graphs edge_weight is None → standard unweighted behavior.
    """
    def __init__(self, node_feat_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)

        # Second GCN layer — commented out until single-layer is verified stable.
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.class_embedding = nn.Embedding(num_classes, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: Data, class_labels: torch.Tensor):
        ew = (data.edge_weight
              if hasattr(data, 'edge_weight') and data.edge_weight is not None
              else None)

        x = F.leaky_relu(
            self.conv1(data.x, data.edge_index, edge_weight=ew), 0.2
        )
        # Uncomment when ready:
        # x = F.leaky_relu(
        #     self.conv2(x, data.edge_index, edge_weight=ew), 0.2
        # )

        x = global_mean_pool(x, data.batch)
        class_embed = self.class_embedding(class_labels.to(x.device))
        x = torch.cat([x, class_embed], dim=1)
        return self.classifier(x)