use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray1, PyArray2, ToPyArray};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;

// -----------------------------------------------------------------------------
//  Data Structures
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Graph {
    num_nodes: usize,
    adj: Vec<Vec<usize>>, // Adjacency list: node -> neighbors
    edges_count: usize,
}

impl Graph {
    fn from_adjacency_matrix(matrix: &[usize], rows: usize, cols: usize) -> Self {
        let mut adj = vec![vec![]; rows];
        let mut edges_count = 0;
        
        for i in 0..rows {
            for j in (i + 1)..cols {
                if matrix[i * cols + j] > 0 {
                    adj[i].push(j);
                    adj[j].push(i);
                    edges_count += 1;
                }
            }
        }
        Graph { num_nodes: rows, adj, edges_count }
    }

    fn to_adjacency_matrix(&self) -> Vec<Vec<usize>> {
        let mut mat = vec![vec![0; self.num_nodes]; self.num_nodes];
        for (u, neighbors) in self.adj.iter().enumerate() {
            for &v in neighbors {
                mat[u][v] = 1;
            }
        }
        mat
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adj[u].contains(&v)
    }

    fn add_edge(&mut self, u: usize, v: usize) -> bool {
        if u == v || self.has_edge(u, v) { return false; }
        self.adj[u].push(v);
        self.adj[v].push(u);
        self.edges_count += 1;
        true
    }

    fn remove_edge(&mut self, u: usize, v: usize) -> bool {
        let pos_u = self.adj[u].iter().position(|&x| x == v);
        let pos_v = self.adj[v].iter().position(|&x| x == u);

        if let (Some(pu), Some(pv)) = (pos_u, pos_v) {
            self.adj[u].swap_remove(pu);
            self.adj[v].swap_remove(pv);
            self.edges_count -= 1;
            return true;
        }
        false
    }
}

// -----------------------------------------------------------------------------
//  Mutations (Paper Implementation)
// -----------------------------------------------------------------------------

impl Graph {
    fn mutate(&mut self, rng: &mut ThreadRng) {
        let ops = [
            "add", "delete", "toggle", 
            "swap", "hop", 
            "local_add", "local_delete", "local_toggle"
        ];
        
        let op = ops.choose(rng).unwrap();
        let n = self.num_nodes;
        if n < 3 { return; }

        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);
        
        match *op {
            "add" => { self.add_edge(u, v); },
            "delete" => { self.remove_edge(u, v); },
            "toggle" => {
                if self.has_edge(u, v) { self.remove_edge(u, v); }
                else { self.add_edge(u, v); }
            },
            "swap" => {
                if self.edges_count < 2 { return; }
                for _ in 0..5 {
                    let u = rng.gen_range(0..n);
                    if self.adj[u].is_empty() { continue; }
                    let v = *self.adj[u].choose(rng).unwrap();
                    
                    let x = rng.gen_range(0..n);
                    if x == u || x == v || self.adj[x].is_empty() { continue; }
                    let y = *self.adj[x].choose(rng).unwrap();
                    if y == u || y == v { continue; }

                    self.remove_edge(u, v);
                    self.remove_edge(x, y);
                    self.add_edge(u, y);
                    self.add_edge(v, x);
                    break;
                }
            },
            "hop" => {
                if self.adj[u].len() < 2 { return; }
                let p = *self.adj[u].choose(rng).unwrap();
                let r = *self.adj[u].choose(rng).unwrap();
                if p == r { return; }

                if !self.has_edge(p, r) {
                    self.remove_edge(p, u);
                    self.add_edge(p, r);
                }
            },
            "local_add" => {
                if self.adj[u].len() < 2 { return; }
                let p = *self.adj[u].choose(rng).unwrap();
                let r = *self.adj[u].choose(rng).unwrap();
                if p != r && !self.has_edge(p, r) {
                    self.add_edge(p, r);
                }
            },
            "local_delete" => {
                if self.adj[u].len() < 2 { return; }
                let p = *self.adj[u].choose(rng).unwrap();
                let r = *self.adj[u].choose(rng).unwrap();
                if p != r && self.has_edge(p, r) {
                    self.remove_edge(p, r);
                }
            },
            "local_toggle" => {
                if self.adj[u].len() < 2 { return; }
                let p = *self.adj[u].choose(rng).unwrap();
                let r = *self.adj[u].choose(rng).unwrap();
                if p != r {
                    if self.has_edge(p, r) { self.remove_edge(p, r); }
                    else { self.add_edge(p, r); }
                }
            }
            _ => {}
        }
    }
}

// -----------------------------------------------------------------------------
//  Statistics & Fitness
// -----------------------------------------------------------------------------

fn compute_histogram(values: &[f32], bins: usize, min_val: f32, max_val: f32) -> Vec<f32> {
    let mut hist = vec![0.0; bins];
    if values.is_empty() { return hist; }
    
    let range = max_val - min_val;
    let step = if range > 0.0 { range / bins as f32 } else { 1.0 };

    for &v in values {
        let bin = ((v - min_val) / step).floor() as isize;
        let idx = bin.clamp(0, (bins - 1) as isize) as usize;
        hist[idx] += 1.0;
    }

    let sum: f32 = hist.iter().sum();
    if sum > 0.0 {
        for x in &mut hist { *x /= sum; }
    }
    hist
}

fn compute_graph_stats(graph: &Graph, num_bins: usize) -> (Vec<f32>, Vec<f32>) {
    let degrees: Vec<f32> = graph.adj.iter()
        .map(|neighbors| neighbors.len() as f32)
        .collect();
    
    let mut clustering = Vec::with_capacity(graph.num_nodes);
    for i in 0..graph.num_nodes {
        let neighbors = &graph.adj[i];
        let k = neighbors.len();
        if k < 2 {
            clustering.push(0.0);
            continue;
        }

        let mut links = 0;
        for idx_a in 0..k {
            for idx_b in (idx_a+1)..k {
                let u = neighbors[idx_a];
                let v = neighbors[idx_b];
                if graph.has_edge(u, v) {
                    links += 1;
                }
            }
        }
        
        let possible_links = (k * (k - 1)) / 2;
        clustering.push(links as f32 / possible_links as f32);
    }

    let deg_hist = compute_histogram(&degrees, num_bins, 0.0, 50.0);
    let clust_hist = compute_histogram(&clustering, num_bins, 0.0, 1.0);

    (deg_hist, clust_hist)
}

fn compute_mse(dist_a: &[f32], dist_b: &[f32]) -> f32 {
    dist_a.iter().zip(dist_b.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

// -----------------------------------------------------------------------------
//  Python Interface
// -----------------------------------------------------------------------------

#[pyfunction]
fn refine_graph(
    py: Python,
    adj_matrix: PyReadonlyArray2<usize>,
    target_degree_hist: PyReadonlyArray1<f32>,
    target_clust_hist: PyReadonlyArray1<f32>,
    generations: usize,
    population_size: usize,
    num_bins: usize
) -> PyResult<Py<PyArray2<usize>>> {
    
    let shape = adj_matrix.shape();
    let rows = shape[0];
    let cols = shape[1];
    let raw_data = adj_matrix.as_slice()?;
    
    let base_graph = Graph::from_adjacency_matrix(raw_data, rows, cols);
    let target_deg = target_degree_hist.as_slice()?;
    let target_clust = target_clust_hist.as_slice()?;

    let mut rng = rand::thread_rng();
    let mut population: Vec<Graph> = (0..population_size)
        .map(|_| {
            let mut g = base_graph.clone();
            for _ in 0..5 { g.mutate(&mut rng); }
            g
        })
        .collect();

    for _gen in 0..generations {
        let mut scored_pop: Vec<(f32, Graph)> = population.par_iter().map(|g| {
            let (d_hist, c_hist) = compute_graph_stats(g, num_bins);
            let score_deg = compute_mse(&d_hist, target_deg);
            let score_clust = compute_mse(&c_hist, target_clust);
            
            let score = score_deg + score_clust;
            (score, g.clone())
        }).collect();

        scored_pop.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let keep_count = std::cmp::max(1, population_size / 5);
        let mut new_pop = Vec::with_capacity(population_size);
        
        for i in 0..keep_count {
            new_pop.push(scored_pop[i].1.clone());
        }

        while new_pop.len() < population_size {
            let parent_idx = rng.gen_range(0..keep_count); 
            let mut child = new_pop[parent_idx].clone();
            
            for _ in 0..2 {
                child.mutate(&mut rng);
            }
            new_pop.push(child);
        }
        population = new_pop;
    }

    let result_adj = population[0].to_adjacency_matrix();
    let py_array = PyArray2::from_vec2(py, &result_adj).unwrap();
    
    Ok(py_array.to_owned())
}

#[pymodule]
fn gwgan_edgerefine_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(refine_graph, m)?)?;
    Ok(())
}