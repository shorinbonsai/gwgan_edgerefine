use crate::graph::GraphState;

pub enum GraphOperation {
    Toggle,
    LocalToggle,
    Hop,
    Add,
    Delete,
    Swap,
    LocalAdd,
    LocalDelete,
}

impl GraphOperation {
    /// Applies a mutation using the pre-decoded vertices (v1..v4) from the genome.
    pub fn apply(&self, graph: &mut GraphState, v1: usize, v2: usize, v3: usize, v4: usize) {
        match self {
            GraphOperation::Toggle => self.apply_toggle(graph, v1, v2),
            GraphOperation::LocalToggle => self.apply_local_toggle(graph, v1, v2, v3),
            GraphOperation::Hop => self.apply_hop(graph, v1, v2, v3),
            GraphOperation::Add => self.apply_add(graph, v1, v2),
            GraphOperation::Delete => self.apply_delete(graph, v1, v2),
            GraphOperation::Swap => self.apply_swap(graph, v1, v2, v3, v4),
            GraphOperation::LocalAdd => self.apply_local_add(graph, v1, v2, v3),
            GraphOperation::LocalDelete => self.apply_local_delete(graph, v1, v2, v3),
        }
    }

  
    fn apply_toggle(&self, graph: &mut GraphState, p: usize, q: usize) {
        if p == q {return;}
        if graph.has_edge(p, q) {
            graph.remove_edge(p, q);
        } else {
            graph.add_edge(p, q);
        }
    }

    fn apply_local_toggle(&self, graph: &mut GraphState, v: usize, n1: usize, n2: usize) {
        // Step 1: Find neighbor 1
        let nb1 = match graph.get_neighbor_at_index(v, n1) {
            Some(nb) => nb,
            None => return,
        };
        // Step 2: Find neighbor 2
        let nb2 = match graph.get_neighbor_at_index(nb1, n2) {
            Some(nb) => nb,
            None => return,
        };
        if v == nb2 {return;} // Avoid self-loop
        // Step 3: Toggle edge between start (v) and end (nb2)
        if graph.has_edge(v, nb2) {
            graph.remove_edge(v, nb2);
        } else {
            graph.add_edge(v, nb2);
        }

    }

    fn apply_hop(&self, graph: &mut GraphState, v: usize, n1: usize, n2: usize) {
        // Logic: If edges (p,q) and (q,r) exist, but (p,r) does not:
        // Remove (p,q) and add (p,r).
        let nb1 = match graph.get_neighbor_at_index(v, n1) {
            Some(nb) => nb,
            None => return,
        };
        if graph.degree(nb1) < 2 {
            return;
        }
        let nb2 = match graph.get_neighbor_at_index(nb1, n2) {
            Some(nb) => nb,
            None => return,
        };
        if v == nb2 {return;} // Avoid self-loop
        //Prevent hop if it closes a triangle(target edge exists)
        if graph.has_edge(v, nb2) {
            return;
        }
        graph.remove_edge(v, nb1);
        graph.add_edge(v, nb2);
    }

    fn apply_add(&self, graph: &mut GraphState, p: usize, q: usize) {
        // Logic: Add edge (p,q) if it doesn't exist.
        if p == q {return;}
        if !graph.has_edge(p, q) {
            graph.add_edge(p, q);
        }
    }

    fn apply_delete(&self, graph: &mut GraphState, p: usize, q: usize) {
        // Logic: Remove edge (p,q) if it exists.
        if p == q {return;}
        if graph.has_edge(p, q) {
            graph.remove_edge(p, q);
        }
    }

    fn apply_swap(&self, graph: &mut GraphState, v1: usize, v2: usize, n1: usize, n2: usize) {
        // Logic: Edge swap (p,q) and (r,s) become (p,s) and (q,r) 
        // effectively preserving degree distribution.
        let min_degree = 2;
        if graph.degree(v1) < min_degree || graph.degree(v2) < min_degree {
            return;
        }
        let nb1 = match graph.get_neighbor_at_index(v1, n1) {
            Some(nb) => nb,
            None => return,
        };
        let nb2 = match graph.get_neighbor_at_index(v2, n2) {
            Some(nb) => nb,
            None => return,
        };
        if v1 == v2 || v1 == nb1 || v1 == nb2 || v2 == nb1 || v2 == nb2 || nb1 == nb2 {
            return; // Avoid self-loops and duplicate edges
        }
        // Check if new edges already exist (was in papers but very strict condition that could lead to extra null operations)
        // if graph.has_edge(v1, nb2) || graph.has_edge(v2, nb1) {
        //     return; // Skip swap if it would create existing edges
        // }
        graph.remove_edge(v1, nb1);
        graph.remove_edge(v2, nb2);
        graph.add_edge(v1, nb2);
        graph.add_edge(v2, nb1);
    }

    fn apply_local_add(&self, graph: &mut GraphState, v: usize, n1: usize, n2: usize) {
        // If (p,q) and (q,r) exist, Add (p,r).
        let nb1 = match graph.get_neighbor_at_index(v, n1) {
            Some(nb) => nb,
            None => return,
        };
        let nb2 = match graph.get_neighbor_at_index(nb1, n2) {
            Some(nb) => nb,
            None => return,
        };
        if v == nb2 {return;} // Avoid self-loop
        // Add (v, nb2) only if it doesn't exist
        if !graph.has_edge(v, nb2) {
            graph.add_edge(v, nb2);
        }
    }

    fn apply_local_delete(&self, graph: &mut GraphState, v: usize, n1: usize, n2: usize) {
        // If (p,q) and (q,r) exist, Delete (p,r).
        let nb1 = match graph.get_neighbor_at_index(v, n1) {
            Some(nb) => nb,
            None => return,
        };
        let nb2 = match graph.get_neighbor_at_index(nb1, n2) {
            Some(nb) => nb,
            None => return,
        };
        if v == nb2 {return;}
        // Delete (v, nb2) only if it exists
        if graph.has_edge(v, nb2) {
            graph.remove_edge(v, nb2);
        }
    }
}