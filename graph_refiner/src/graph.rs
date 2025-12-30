/// A lightweight graph representation optimized for the specific
/// mutation operations defined in the paper.
#[derive(Clone)]
pub struct GraphState {
    num_nodes: usize,
    // Adjacency list might be better for 'Hop' and 'Local' operations
    // which require checking neighbors of neighbors.
    adjacency: Vec<Vec<usize>>, 
    pub fitness: f64,
}

impl GraphState {
    pub fn new(num_nodes: usize) -> Self {
        GraphState {
            num_nodes,
            adjacency: vec![vec![]; num_nodes],
            fitness: 0.0,
        }
    }

    pub fn set_edges(&mut self, edges: &[(usize, usize)]) {
        for (u, v) in edges {
            self.add_edge(*u, *v);
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u < self.num_nodes && v < self.num_nodes {
             if !self.adjacency[u].contains(&v) {
                 self.adjacency[u].push(v);
                 self.adjacency[v].push(u); // Undirected
             }
        }
    }

    pub fn remove_edge(&mut self, u: usize, v: usize) {
        if let Some(pos) = self.adjacency[u].iter().position(|&x| x == v) {
            self.adjacency[u].remove(pos);
        }
        if let Some(pos) = self.adjacency[v].iter().position(|&x| x == u) {
            self.adjacency[v].remove(pos);
        }
    }

    pub fn get_edge_list(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for (u, neighbors) in self.adjacency.iter().enumerate() {
            for &v in neighbors {
                if u < v { // Avoid duplicates for undirected graph
                    edges.push((u, v));
                }
            }
        }
        edges
    }
}