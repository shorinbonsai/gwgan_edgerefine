use rand::{Rng, SeedableRng};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha8Rng;
use crate::stats::{degree_distribution, clustering_distribution, spectral_features, compute_mmd};
use rayon::prelude::*;

use crate::graph::GraphState;
use crate::operations::GraphOperation;

/// A genetic optimizer that represents each individual as a command string.
///
/// Instead of storing a graph directly for each individual, each genome is a
/// vector of encoded commands.  Each command encodes both the THADS‑N
/// operation to perform and a free parameter.  During `initialize_population`
/// the genomes are randomly generated.  The `express` method applies the
/// sequence of commands to a base graph to produce a phenotype graph. 
pub struct GeneticOptimizer {
    /// The population of genomes.  Each genome is a vector of u64 encoded
    /// commands.  The lower three bits encode the operator (0‒7) and the
    /// upper bits can store parameters for that operator.
    population: Vec<Vec<u64>>,
    /// Number of genomes in the population.
    population_size: usize,
    /// Length of each genome (number of commands).
    gene_length: usize,
    /// The initial graph from the WGAN.  Genomes are expressed relative to
    /// this base graph.  Stored so that `get_best_edges` can reconstruct the
    /// refined graph.
    base_graph: Option<GraphState>,
    /// The best genome discovered so far.
    best_genome: Option<Vec<u64>>,
    /// Target statistics for fitness evaluation.
    target_degrees: Vec<Vec<f64>>,
    degree_mean: Vec<f64>,
    degree_std: Vec<f64>,
    gamma_degree: f64,

    target_clustering: Vec<Vec<f64>>,
    clustering_mean: Vec<f64>,
    clustering_std: Vec<f64>,
    gamma_clustering: f64,

    target_spectral: Vec<Vec<f64>>,
    spectral_mean: Vec<f64>,
    spectral_std: Vec<f64>,
    gamma_spectral: f64,
    weights: (f64, f64, f64), // (degree, clustering, spectral)
    op_weights: Vec<f64>,
}

impl GeneticOptimizer {
    /// Create a new genetic optimizer.  `population_size` sets the number of
    /// genomes maintained and `gene_length` sets the number of commands in
    /// each genome.  The base graph is initialized when
    /// `initialize_population` is called.
    pub fn new(population_size: usize, gene_length: usize) -> Self {
        GeneticOptimizer {
            population: Vec::new(),
            population_size,
            gene_length,
            base_graph: None,
            best_genome: None,
            target_degrees: Vec::new(),
            degree_mean: Vec::new(),
            degree_std: Vec::new(),
            gamma_degree: 1.0, // Default to 1.0

            target_clustering: Vec::new(),
            clustering_mean: Vec::new(),
            clustering_std: Vec::new(),
            gamma_clustering: 1.0,

            target_spectral: Vec::new(),
            spectral_mean: Vec::new(),
            spectral_std: Vec::new(),
            gamma_spectral: 1.0,

            weights: (0.0, 0.0, 0.0),
            op_weights: vec![1.0; 9], // Equal weights for 9 operations
        }
    }

    /// Set the weights for the operators (0..8).
    /// This should be called from Python immediately after creation.
    pub fn set_op_weights(&mut self, weights: Vec<f64>) {
        if weights.len() != 9 {
             panic!("Must provide exactly 9 weights (0-7 Ops + 8 Null)");
        }
        self.op_weights = weights;
    }

    fn generate_gene<R: Rng>(rng: &mut R, dist: &WeightedIndex<f64>) -> u64 {
        let op: u64 = dist.sample(rng) as u64;
        let param: u64 = rng.random::<u32>() as u64;
        (param << 4) | op
    }

    /// Initialize the population.  This method takes the number of nodes and
    /// the edge list of the initial graph produced by the WGAN.  It sets
    /// `base_graph` accordingly and generates `population_size` random
    /// genomes.  Each command in a genome is formed by combining a random
    /// operator (0‒7) with a random 32‑bit parameter encoded in the upper
    /// bits.  The best genome is initialized to the first genome.
    pub fn initialize_population(&mut self, num_nodes: usize, initial_edges: Vec<(usize, usize)>, seed: u64) {
        // Build the base graph used as the starting point for all genomes.
        let mut base = GraphState::new(num_nodes);
        base.set_edges(&initial_edges);
        self.base_graph = Some(base);

        // Seed the RNG deterministically
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Create the weighted distribution based on input densities
        let dist = WeightedIndex::new(&self.op_weights)
            .expect("Invalid weights provided (e.g., all zero)");

        self.population.clear();
        for _ in 0..self.population_size {
            let mut genome = Vec::with_capacity(self.gene_length);
            for _ in 0..self.gene_length {
                genome.push(Self::generate_gene(&mut rng, &dist));
            }
            self.population.push(genome);
        }
        // Set the first genome as the current best if any exist.
        if let Some(genome) = self.population.first() {
            self.best_genome = Some(genome.clone());
        }
    }

    pub fn set_targets(
        &mut self,
        target_degrees: Vec<Vec<f64>>, degree_mean: Vec<f64>, degree_std: Vec<f64>,
        target_clustering: Vec<Vec<f64>>, clustering_mean: Vec<f64>, clustering_std: Vec<f64>,
        target_spectral: Vec<Vec<f64>>, spectral_mean: Vec<f64>, spectral_std: Vec<f64>,
        weights: (f64, f64, f64),
        gammas: (f64, f64, f64)
    ) {
        self.target_degrees = target_degrees;
        self.degree_mean = degree_mean;
        self.degree_std = degree_std;
        
        self.target_clustering = target_clustering;
        self.clustering_mean = clustering_mean;
        self.clustering_std = clustering_std;

        self.target_spectral = target_spectral;
        self.spectral_mean = spectral_mean;
        self.spectral_std = spectral_std;

        self.weights = weights;
        
        // Store gammas directly
        self.gamma_degree = gammas.0;
        self.gamma_clustering = gammas.1;
        self.gamma_spectral = gammas.2;
    }

    /// Decode and apply the commands in `genome` to `base_graph` to produce
    /// a new graph.  The lower three bits of each command determine the
    /// operation while the remaining bits can encode parameters.   The resulting graph is
    /// returned.
    fn express(&self, genome: &[u64], base_graph: &GraphState) -> GraphState {
        let mut graph = base_graph.clone();
        let num_nodes = base_graph.num_nodes;

        for &gene in genome {
            let op_code: u8 = (gene & 0xF) as u8;
            let param_payload = gene >> 4;
            // Decode 4 potential vertices (v1, v2, v3, v4) from the payload.
            // We use modulo arithmetic to ensure they are ALWAYS valid indices [0, num_nodes).
            // This replicates the 'block % verts' safety logic from C++ 'express'.
            // Note: u64 has 64 bits. 3 bits for op = 61 bits payload.
            // Even for 1 million nodes, we can easily fit 3-4 vertices.
            let v1 = (param_payload % num_nodes as u64) as usize;
            let v2 = ((param_payload / num_nodes as u64) % num_nodes as u64) as usize;
            //Powers of num_nodes for v3 and v4
            let v3 = ((param_payload / (num_nodes as u64).pow(2)) % num_nodes as u64) as usize;
            let v4 = ((param_payload / (num_nodes as u64).pow(3)) % num_nodes as u64) as usize;
            

            // Map op_code into an operation.  If the op_code is out of
            // range we skip the command.
            let operation = match op_code {
                0 => Some(GraphOperation::Toggle),
                1 => Some(GraphOperation::LocalToggle),
                2 => Some(GraphOperation::Hop),
                3 => Some(GraphOperation::Add),
                4 => Some(GraphOperation::Delete),
                5 => Some(GraphOperation::Swap),
                6 => Some(GraphOperation::LocalAdd),
                7 => Some(GraphOperation::LocalDelete),
                8 => Some(GraphOperation::Null),
                _ => None,
            };
            if let Some(op) = operation {
                op.apply(&mut graph, v1, v2, v3, v4);
            }
        }
        graph
    }

    /// Calculates fitness (Error Score).
    /// Returns a positive float where LOWER is BETTER.
    fn calculate_fitness(&self, genome: &[u64], base: &GraphState) -> f64 {
        let graph = self.express(genome, base);
        
        let deg = crate::stats::degree_distribution(&graph, 10);
        let clust = crate::stats::clustering_distribution(&graph, 10);
        let spec = crate::stats::spectral_features(&graph, 10);

        // MMD is a distance/divergence. Smaller is better.
        let score_deg = crate::stats::compute_mmd(&deg, &self.target_degrees, &self.degree_mean, &self.degree_std, self.gamma_degree);
        let score_clust = crate::stats::compute_mmd(&clust, &self.target_clustering, &self.clustering_mean, &self.clustering_std, self.gamma_clustering);
        let score_spec = crate::stats::compute_mmd(&spec, &self.target_spectral, &self.spectral_mean, &self.spectral_std, self.gamma_spectral);

        // Weighted Sum of Errors
        (score_deg * self.weights.0) + (score_clust * self.weights.1) + (score_spec * self.weights.2)
    }

    /// Tournament Selection: Picks `k` random individuals and returns the best one.
    /// MINIMIZATION: Returns the individual with the LOWEST fitness score.
    fn tournament_select<R: Rng>(
        population: &[Vec<u64>], 
        fitnesses: &[f64], 
        k: usize, 
        rng: &mut R
    ) -> Vec<u64> {
        let mut best_idx = 0;
        let mut best_fit = f64::INFINITY;
        
        for _ in 0..k {
            let idx = rng.random_range(0..population.len());
            if fitnesses[idx] < best_fit {
                best_fit = fitnesses[idx];
                best_idx = idx;
            }
        }
        population[best_idx].clone()
    }

    /// Two-Point Crossover: Swaps segments between parents.
    fn crossover<R: Rng>(p1: &[u64], p2: &[u64], rng: &mut R) -> (Vec<u64>, Vec<u64>) {
        let len = p1.len();
        let mut c1 = p1.to_vec();
        let mut c2 = p2.to_vec();

        if len < 2 { return (c1, c2); }

        let pt1 = rng.random_range(0..len);
        let pt2 = rng.random_range(0..len);
        let (start, end) = if pt1 < pt2 { (pt1, pt2) } else { (pt2, pt1) };

        for i in start..end {
            c1[i] = p2[i];
            c2[i] = p1[i];
        }
        (c1, c2)
    }

    /// Mutation: Replaces 1 to MNM genes with new random ones.
    fn mutate<R: Rng>(genome: &mut Vec<u64>, rng: &mut R, dist: &WeightedIndex<f64>) {
        let mnm = 4; // Max mutations, matching C++
        let num_mutations = rng.random_range(1..=mnm);
        
        for _ in 0..num_mutations {
            let idx = rng.random_range(0..genome.len());
            genome[idx] = Self::generate_gene(rng, dist);
        }
    }

    pub fn evolve(&mut self, generations: usize, seed: u64) -> f64 {
        if self.base_graph.is_none() { return f64::INFINITY; }
        let base = self.base_graph.as_ref().unwrap();

        let tournament_size = 5; 
        let mut best_overall_error = f64::INFINITY; 
        let op_weights = self.op_weights.clone();

        for g in 0..generations {
            // 1. Evaluate Fitness
            let fitness_results: Vec<(usize, f64)> = self.population.par_iter().enumerate().map(|(i, genome)| {
                let score = self.calculate_fitness(genome, base);
                (i, score)
            }).collect();

            // 2. Process Results (Minimization)
            let mut fitnesses = vec![0.0; self.population.len()];
            let mut min_gen_error = f64::INFINITY;
            let mut best_gen_idx = 0;

            for (i, err) in fitness_results {
                fitnesses[i] = err;
                if err < min_gen_error {
                    min_gen_error = err;
                    best_gen_idx = i;
                }
            }

            if min_gen_error < best_overall_error {
                best_overall_error = min_gen_error;
                self.best_genome = Some(self.population[best_gen_idx].clone());
            }

            // 3. Selection & Reproduction
            let old_pop = &self.population; 
            
            let new_population: Vec<Vec<u64>> = (0..self.population_size / 2)
                .into_par_iter()
                .map(|i| {
                    let task_seed = seed
                        .wrapping_add((g as u64).wrapping_mul(1_000_000))
                        .wrapping_add(i as u64);
                    
                    let mut rng = ChaCha8Rng::seed_from_u64(task_seed);
                    let dist = WeightedIndex::new(&op_weights).unwrap();

                    let p1 = Self::tournament_select(old_pop, &fitnesses, tournament_size, &mut rng);
                    let p2 = Self::tournament_select(old_pop, &fitnesses, tournament_size, &mut rng);

                    let (mut c1, mut c2) = Self::crossover(&p1, &p2, &mut rng);

                    Self::mutate(&mut c1, &mut rng, &dist);
                    Self::mutate(&mut c2, &mut rng, &dist);

                    vec![c1, c2]
                })
                .flatten()
                .collect();

            self.population = new_population;

            // 4. Elitism
            if self.population.len() > self.population_size {
                self.population.truncate(self.population_size);
            }
            if let Some(ref best) = self.best_genome {
                self.population[0] = best.clone();
            }
        }

        best_overall_error
    }

    /// Reconstruct the best graph discovered so far and return its edge list.
    /// If no base graph or best genome exists, returns an empty vector.
    pub fn get_best_edges(&self) -> Vec<(usize, usize)> {
        if let (Some(ref genome), Some(ref base)) = (self.best_genome.as_ref(), self.base_graph.as_ref()) {
            let graph = self.express(genome, base);
            return graph.get_edge_list();
        }
        Vec::new()
    }
}