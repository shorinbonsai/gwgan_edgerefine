use rand::Rng;
use rayon::prelude::*;

use crate::graph::GraphState;
use crate::operations::GraphOperation;

/// A genetic optimizer that represents each individual as a command string.
///
/// Instead of storing a graph directly for each individual, each genome is a
/// vector of encoded commands.  Each command encodes both the THADS‑N
/// operation to perform and a free parameter.  During `initialize_population`
/// the genomes are randomly generated.  The `express` method applies the
/// sequence of commands to a base graph to produce a phenotype graph.  For
/// now selection, crossover and mutation are not implemented; the algorithm
/// simply evaluates the genomes by expressing them and keeps track of the
/// best genome seen.
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
        }
    }

    /// Initialize the population.  This method takes the number of nodes and
    /// the edge list of the initial graph produced by the WGAN.  It sets
    /// `base_graph` accordingly and generates `population_size` random
    /// genomes.  Each command in a genome is formed by combining a random
    /// operator (0‒7) with a random 32‑bit parameter encoded in the upper
    /// bits.  The best genome is initialized to the first genome.
    pub fn initialize_population(&mut self, num_nodes: usize, initial_edges: Vec<(usize, usize)>) {
        // Build the base graph used as the starting point for all genomes.
        let mut base = GraphState::new(num_nodes);
        base.set_edges(&initial_edges);
        self.base_graph = Some(base);

        // Generate random genomes.
        use rand::thread_rng;
        let mut rng = thread_rng();
        self.population.clear();
        for _ in 0..self.population_size {
            let mut genome = Vec::with_capacity(self.gene_length);
            for _ in 0..self.gene_length {
                let op: u8 = rng.gen_range(0..8);
                let param: u64 = rng.gen::<u32>() as u64;
                let cmd: u64 = (param << 3) | (op as u64);
                genome.push(cmd);
            }
            self.population.push(genome);
        }
        // Set the first genome as the current best if any exist.
        if let Some(genome) = self.population.first() {
            self.best_genome = Some(genome.clone());
        }
    }

    /// Decode and apply the commands in `genome` to `base_graph` to produce
    /// a new graph.  The lower three bits of each command determine the
    /// operation while the remaining bits can encode parameters.   The resulting graph is
    /// returned.
    fn express(&self, genome: &[u64], base_graph: &GraphState) -> GraphState {
        let mut graph = base_graph.clone();
        let num_nodes = base_graph.num_nodes;

        for &gene in genome {
            let op_code: u8 = (gene & 0b111) as u8;
            let param_payload = gene >> 3;
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
                _ => None,
            };
            if let Some(op) = operation {
                op.apply(&mut graph, v1, v2, v3, v4);
            }
        }
        graph
    }

    /// Run the genetic algorithm for a given number of generations.  The
    /// current implementation simply expresses the genomes relative to the
    /// base graph and records the first genome as the best.  Parallelism is
    /// available via Rayon, but no selection, crossover or mutation is
    /// performed.  Returns `0.0` as a placeholder fitness score.
    pub fn evolve(&mut self, generations: usize) -> f64 {
        let _ = generations; // generations are unused in the stub
        // Evaluate all genomes by expressing them.  In a complete
        // implementation you would compute a fitness score here.
        if let Some(ref base) = self.base_graph {
            self.population.par_iter().for_each(|genome| {
                let _graph = self.express(genome, base);
                // compute fitness here
            });
            // Choose the first genome as the best for now.
            if let Some(genome) = self.population.first() {
                self.best_genome = Some(genome.clone());
            }
        }
        0.0
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