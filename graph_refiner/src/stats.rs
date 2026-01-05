use crate::graph::GraphState;
use nalgebra::{DMatrix, SymmetricEigen};

/// Compute the degree distribution histogram of the graph.
/// Matches Python `utils.py` behavior: range is adaptive per graph.
pub fn degree_distribution(graph: &GraphState, num_bins: usize) -> Vec<f64> {
    if graph.num_nodes ==0 {return vec![0.0; num_bins];}
    let degrees: Vec<usize> = (0..graph.num_nodes).map(|n| graph.degree(n)).collect();
    let local_max_degree = *degrees.iter().max().unwrap_or(&0);

    let mut hist = vec![0.0; num_bins];

    // In Python utils.py: range=(0, max(int(deg.max()), num_bins))
    let range_max = std::cmp::max(local_max_degree, num_bins);

    for &deg in &degrees {
        // Map degree 'd' to a bin index.
        // Logic: bin_idx = floor(deg / (range_max / num_bins))
        let bin_idx = if range_max == 0 {
            0
        } else {
            std::cmp::min(deg * num_bins / range_max, num_bins - 1)
        };
        hist[bin_idx] += 1.0;
    }
    // Normalize to probability distribution.
    let sum: f64 = hist.iter().sum();
    if sum > 0.0 {
        for x in &mut hist { *x /= sum; }
    }
    hist
}

/// Computes the clustering coefficient distribution.
pub fn clustering_distribution(graph: &GraphState, num_bins: usize) -> Vec<f64> {
    if graph.num_nodes < 3 {return vec![0.0; num_bins];}

    let mut coeffs = Vec::with_capacity(graph.num_nodes);
    for u in 0..graph.num_nodes {
        let neighbors = &graph.adjacency[u];
        let k = neighbors.len();
        if k < 2 {
            coeffs.push(0.0);
            continue;
        }
        let mut links = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                if graph.has_edge(neighbors[i], neighbors[j]) {
                    links += 1;
                }
            }
        }
        coeffs.push((2.0 * links as f64) / (k * (k - 1)) as f64);
    }
    let mut hist = vec![0.0; num_bins];
    for &c in &coeffs {
        // Range is [0.0, 1.0]
        let bin_idx = std::cmp::min((c * (num_bins as f64)) as usize, num_bins - 1);
        hist[bin_idx] += 1.0;
    }

    let sum: f64 = hist.iter().sum();
    if sum > 0.0 {
        for x in &mut hist { *x /= sum; }
    }
    hist
}

/// Computes spectral features (top-k eigenvalues).
/// `num_features`: Explicitly passed, no magic numbers.
pub fn spectral_features(graph: &GraphState, num_features: usize) -> Vec<f64> {
    let n = graph.num_nodes;
    if n < 2 { return vec![0.0; num_features]; }

    // 1. Build Laplacian (L = D - A)
    let mut laplacian = DMatrix::<f64>::zeros(n, n);
    for u in 0..n {
        laplacian[(u, u)] = graph.degree(u) as f64;
        for &v in &graph.adjacency[u] {
            laplacian[(u, v)] = -1.0;
        }
    }

    // 2. Compute Eigenvalues
    // SymmetricEigen is efficient for this
    let eig = SymmetricEigen::new(laplacian);
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.into_iter().copied().collect();

    // 3. Sort
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // 4. Pad or Truncate
    if eigenvalues.len() < num_features {
        eigenvalues.resize(num_features, 0.0);
    } else {
        eigenvalues.truncate(num_features);
    }
    eigenvalues
}

pub fn compute_mmd(
    candidate: &[f64], 
    targets_normalized: &[Vec<f64>], 
    target_mean: &[f64], 
    target_std: &[f64],
    sigma: f64
) -> f64 {
    if targets_normalized.is_empty() { return 0.0; }
    
    // Normalize candidate using the TARGET's stats
    let candidate_norm: Vec<f64> = candidate.iter()
        .zip(target_mean.iter())
        .zip(target_std.iter())
        .map(|((&val, &mu), &sigma)| (val - mu) / sigma)
        .collect();

    let n = targets_normalized.len();
    let gamma = 1.0 / (2.0 * sigma * sigma); 
    
    let mut sum_k_xy = 0.0;
    for target in targets_normalized {
        let dist_sq: f64 = candidate_norm.iter().zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        sum_k_xy += (-gamma * dist_sq).exp();
    }
    
    // Return negative similarity (minimization objective)
    -(sum_k_xy / n as f64)
}