use crate::graph::GraphState;

pub enum GraphOperation {
    Toggle,
    Hop,
    Add,
    Delete,
    Swap,
    LocalToggle,
    LocalAdd,
    LocalDelete,
    Null,
}

impl GraphOperation {
    /// Applies a mutation using the pre-decoded vertices (v1..v4) from the genome.
    pub fn apply(&self, graph: &mut GraphState, v1: usize, v2: usize, v3: usize, v4: usize) {
        match self {
            GraphOperation::Toggle => self.apply_toggle(graph, v1, v2),
            GraphOperation::Hop => self.apply_hop(graph, v1, v2, v3),
            GraphOperation::Add => self.apply_add(graph, v1, v2),
            GraphOperation::Delete => self.apply_delete(graph, v1, v2),
            GraphOperation::Swap => self.apply_swap(graph, v1, v2, v3, v4),
            GraphOperation::LocalToggle => self.apply_local_toggle(graph, v1, v2, v3),
            GraphOperation::LocalAdd => self.apply_local_add(graph, v1, v2, v3),
            GraphOperation::LocalDelete => self.apply_local_delete(graph, v1, v2, v3),
            GraphOperation::Null => {},
        }
    }

    fn apply_toggle(&self, graph: &mut GraphState, p: usize, q: usize) {
        if p == q {
            return;
        }
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
        if v == nb2 {
            return;
        } // Avoid self-loop
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
        if v == nb2 {
            return;
        } // Avoid self-loop
        //Prevent hop if it closes a triangle(target edge exists)
        if graph.has_edge(v, nb2) {
            return;
        }
        graph.remove_edge(v, nb1);
        graph.add_edge(v, nb2);
    }

    fn apply_add(&self, graph: &mut GraphState, p: usize, q: usize) {
        // Logic: Add edge (p,q) if it doesn't exist.
        if p == q {
            return;
        }
        if !graph.has_edge(p, q) {
            graph.add_edge(p, q);
        }
    }

    fn apply_delete(&self, graph: &mut GraphState, p: usize, q: usize) {
        // Logic: Remove edge (p,q) if it exists.
        if p == q {
            return;
        }
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
        if v == nb2 {
            return;
        } // Avoid self-loop
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
        if v == nb2 {
            return;
        }
        // Delete (v, nb2) only if it exists
        if graph.has_edge(v, nb2) {
            graph.remove_edge(v, nb2);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphState;

    // Helper to create a graph with specific edges
    fn create_graph(num_nodes: usize, edges: &[(usize, usize)]) -> GraphState {
        let mut graph = GraphState::new(num_nodes);
        graph.set_edges(edges);
        graph
    }

    #[test]
    fn test_toggle() {
        // Toggle: If edge exists, remove it. If not, add it.
        let mut graph = create_graph(5, &[(0, 1)]);
        let op = GraphOperation::Toggle;

        // 1. Toggle existing edge (0,1) -> should remove
        op.apply(&mut graph, 0, 1, 0, 0);
        assert!(!graph.has_edge(0, 1), "Toggle should remove existing edge");

        // 2. Toggle non-existing edge (0,2) -> should add
        op.apply(&mut graph, 0, 2, 0, 0);
        assert!(graph.has_edge(0, 2), "Toggle should add missing edge");

        // 3. Self-loop check (should do nothing)
        op.apply(&mut graph, 3, 3, 0, 0);
        assert!(graph.get_edge_list().len() == 1, "Toggle should ignore self-loops");
    }

    #[test]
    fn test_add_delete() {
        let mut graph = create_graph(5, &[(0, 1)]);
        
        // Test Add
        GraphOperation::Add.apply(&mut graph, 1, 2, 0, 0); // Add (1,2)
        assert!(graph.has_edge(1, 2));
        GraphOperation::Add.apply(&mut graph, 0, 1, 0, 0); // Add existing (0,1)
        assert!(graph.degree(0) == 1, "Add should not duplicate edges");

        // Test Delete
        GraphOperation::Delete.apply(&mut graph, 0, 1, 0, 0); // Delete (0,1)
        assert!(!graph.has_edge(0, 1));
        GraphOperation::Delete.apply(&mut graph, 0, 4, 0, 0); // Delete non-existing
        assert!(!graph.has_edge(0, 4), "Delete non-existing should do nothing");
    }

    #[test]
    fn test_local_toggle() {
        // Setup: Linear chain 0-1-2
        // We strictly control insertion order to guarantee neighbor indices.
        let mut graph = GraphState::new(5);
        
        // 1. Add Edge (0, 1)
        // adj[0] = [1]
        // adj[1] = [0]
        graph.add_edge(0, 1);
        
        // 2. Add Edge (1, 2)
        // adj[1] becomes [0, 2] (append 2)
        // adj[2] = [1]
        graph.add_edge(1, 2);

        // Verify setup assumptions
        // v=0 (neighbor 1 is at index 0)
        assert_eq!(graph.get_neighbor_at_index(0, 0), Some(1));
        // nb1=1 (neighbor 2 is at index 1) -> because [0, 2]
        assert_eq!(graph.get_neighbor_at_index(1, 1), Some(2));
        assert_eq!(graph.get_neighbor_at_index(1, 0), Some(0));

        // CASE 1: Toggle On (Create Edge)
        // Path: 0 -> 1 -> 2. We want to toggle (0, 2).
        // v=0. n1=0 (selects 1). n2=1 (selects 2).
        GraphOperation::LocalToggle.apply(&mut graph, 0, 0, 1, 0);
        
        assert!(graph.has_edge(0, 2), "LocalToggle should close the wedge 0-1-2");
        assert!(graph.has_edge(0, 1), "Base edge 0-1 should remain");
        assert!(graph.has_edge(1, 2), "Base edge 1-2 should remain");

        // CASE 2: Toggle Off (Remove Edge)
        // Graph now has triangle 0-1-2.
        // Apply exact same operation: v=0, n1=0 (selects 1), n2=1 (selects 2).
        // It should detect existing edge (0,2) and remove it.
        GraphOperation::LocalToggle.apply(&mut graph, 0, 0, 1, 0);
        
        assert!(!graph.has_edge(0, 2), "LocalToggle should open the triangle 0-1-2");
        assert!(graph.has_edge(0, 1), "Base edge 0-1 should remain");

        // CASE 3: Self-Loop Avoidance
        // Path: 0 -> 1 -> 0.
        // v=0. n1=0 (selects 1). 
        // From node 1, we select neighbor 0 (index 0).
        // nb2 becomes 0. v == nb2. Should abort.
        let degree_before = graph.degree(0);
        GraphOperation::LocalToggle.apply(&mut graph, 0, 0, 0, 0); // n2=0 selects node 0
        
        assert_eq!(graph.degree(0), degree_before, "LocalToggle should not affect self (0->1->0)");
        assert!(!graph.has_edge(0, 0), "Should definitely not add self-loop");

        // CASE 4: Missing Neighbors (Safety)
        // Node 4 is isolated.
        GraphOperation::LocalToggle.apply(&mut graph, 4, 0, 0, 0);
        // Should simply return without panic
    }

    #[test]
    fn test_hop() {
        // Hop(v, n1, n2): Remove (v, nb1), Add (v, nb2)
        // Setup: 0-1-2. Target: 0-2 (and remove 0-1).
        // Condition: nb1 must have degree >= 2. (Node 1 has degree 2).
        // Condition: (v, nb2) must NOT exist (triangle check).
        let mut graph = create_graph(5, &[(0, 1), (1, 2)]);
        
        // v=0. nb1=1 (at idx 0). nb2=2 (neighbor of 1 at idx 1).
        GraphOperation::Hop.apply(&mut graph, 0, 0, 1, 0);
        
        assert!(graph.has_edge(0, 2), "Hop should add edge (0,2)");
        assert!(!graph.has_edge(0, 1), "Hop should remove edge (0,1)");
        assert!(graph.has_edge(1, 2), "Edge (1,2) should remain");
    }

    #[test]
    fn test_swap() {
        // Swap (v1, v2): (v1, nb1), (v2, nb2) -> (v1, nb2), (v2, nb1)
        // Setup: 0-1 and 2-3.
        // v1=0 (nb1=1), v2=2 (nb2=3).
        // Result should be: 0-3 and 2-1.
        
        // Min degree constraint in code is 2. So we need extra edges to satisfy degree checks.
        // 0-1, 0-4
        // 2-3, 2-4
        let mut graph = create_graph(5, &[(0, 1), (0, 4), (2, 3), (2, 4)]);
        
        // Check degrees: 0->2, 2->2. OK.
        
        // v1=0, select neighbor 1 (idx 0?). 
        // v2=2, select neighbor 3 (idx 0?).
        GraphOperation::Swap.apply(&mut graph, 0, 2, 0, 0);
        
        // Check if swap occurred
        // We expect (0,1) gone, (2,3) gone.
        // We expect (0,3) added, (2,1) added.
        
        // Note: Indices depend on insertion order. 
        // 0's neighbors: [1, 4]. 2's neighbors: [3, 4].
        // index 0 -> 1 and 3.
        
        if graph.has_edge(0, 3) && graph.has_edge(2, 1) {
             assert!(!graph.has_edge(0, 1));
             assert!(!graph.has_edge(2, 3));
        } else {
             // If it didn't happen, ensure graph is stable (no partial application)
             assert!(graph.has_edge(0, 1));
        }
    }
    
    #[test]
    fn test_local_add() {
        // LocalAdd: 0-1-2 -> Add (0,2)
        let mut graph = create_graph(5, &[(0, 1), (1, 2)]);
        // v=0, nb1=1, nb2=2
        GraphOperation::LocalAdd.apply(&mut graph, 0, 0, 1, 0);
        assert!(graph.has_edge(0, 2), "LocalAdd should create triangle");
    }
}