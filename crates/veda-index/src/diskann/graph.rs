use serde::{Deserialize, Serialize};

/// Vamana graph: simple adjacency list with a medoid entry point.
#[derive(Debug, Serialize, Deserialize)]
pub struct VamanaGraph {
    adjacency: Vec<Vec<u32>>,
    medoid: Option<u32>,
}

impl VamanaGraph {
    /// Create a new graph with `n` nodes and empty adjacency lists.
    pub fn new(n: usize) -> Self {
        Self {
            adjacency: vec![Vec::new(); n],
            medoid: None,
        }
    }

    /// Add a directed edge from `from` to `to` (does not check duplicates).
    pub fn add_edge(&mut self, from: u32, to: u32) {
        self.adjacency[from as usize].push(to);
    }

    /// Replace the entire neighbor list of `id`.
    pub fn set_neighbors(&mut self, id: u32, neighbors: Vec<u32>) {
        self.adjacency[id as usize] = neighbors;
    }

    /// Get the neighbor list of `id`.
    pub fn get_neighbors(&self, id: u32) -> &[u32] {
        &self.adjacency[id as usize]
    }

    /// Set the medoid (entry point).
    pub fn set_medoid(&mut self, id: u32) {
        self.medoid = Some(id);
    }

    /// Get the medoid.
    pub fn medoid(&self) -> Option<u32> {
        self.medoid
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Returns `true` if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }
}
