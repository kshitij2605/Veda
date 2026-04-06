use serde::{Deserialize, Serialize};

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// The unique identifier for this node (matches the vector ID).
    pub id: u32,
    /// The level this node was assigned during insertion.
    pub level: usize,
    /// Neighbors per layer: `neighbors[layer]` is the list of neighbor IDs at that layer.
    pub neighbors: Vec<Vec<u32>>,
}

/// The HNSW graph structure holding all nodes and the entry point.
#[derive(Debug, Serialize, Deserialize)]
pub struct HnswGraph {
    nodes: Vec<Option<HnswNode>>,
    entry_point: Option<u32>,
    max_level: usize,
}

impl HnswGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
        }
    }

    /// Add a node with the given ID and level. Initializes empty neighbor lists
    /// for each layer from 0 to `level` (inclusive).
    pub fn add_node(&mut self, id: u32, level: usize) -> &mut HnswNode {
        let idx = id as usize;
        if idx >= self.nodes.len() {
            self.nodes.resize_with(idx + 1, || None);
        }
        let neighbors = (0..=level).map(|_| Vec::new()).collect();
        self.nodes[idx] = Some(HnswNode {
            id,
            level,
            neighbors,
        });
        self.nodes[idx].as_mut().unwrap()
    }

    /// Get a reference to the node with the given ID, if it exists.
    pub fn get_node(&self, id: u32) -> Option<&HnswNode> {
        self.nodes.get(id as usize).and_then(|n| n.as_ref())
    }

    /// Get the neighbor list for the given node at the specified layer.
    /// Returns an empty slice if the node doesn't exist or doesn't have that layer.
    pub fn get_neighbors(&self, id: u32, layer: usize) -> &[u32] {
        match self.get_node(id) {
            Some(node) if layer < node.neighbors.len() => &node.neighbors[layer],
            _ => &[],
        }
    }

    /// Set the neighbor list for the given node at the specified layer.
    pub fn set_neighbors(&mut self, id: u32, layer: usize, neighbors: Vec<u32>) {
        let idx = id as usize;
        if let Some(Some(node)) = self.nodes.get_mut(idx) {
            if layer < node.neighbors.len() {
                node.neighbors[layer] = neighbors;
            }
        }
    }

    /// Get the current entry point of the graph.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Set the entry point of the graph.
    pub fn set_entry_point(&mut self, id: u32) {
        self.entry_point = Some(id);
    }

    /// Get the current maximum level in the graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Set the maximum level of the graph.
    pub fn set_max_level(&mut self, level: usize) {
        self.max_level = level;
    }

    /// Get the total number of nodes in the graph (including None slots).
    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }
}
