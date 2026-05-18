//! Directed Acyclic Graph (DAG) resolution for task dependencies.
//!
//! This module provides utilities for working with task dependency graphs,
//! including topological sorting and cycle detection.

use std::collections::{HashMap, HashSet};

/// Error types for DAG operations.
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    /// A cycle was detected in the dependency graph.
    #[error("Cycle detected in dependency graph: {0}")]
    Cycle(String),

    /// A task depends on a non-existent task.
    #[error("Task '{task}' depends on unknown task '{dependency}'")]
    UnknownDependency {
        task: String,
        dependency: String,
    },
}

/// A node in the dependency graph.
#[derive(Debug, Clone)]
pub struct DagNode<T> {
    /// The value associated with this node.
    pub value: T,

    /// Names of tasks this task depends on.
    pub dependencies: Vec<String>,
}

impl<T> DagNode<T> {
    /// Create a new DAG node.
    pub fn new(value: T, dependencies: Vec<String>) -> Self {
        Self {
            value,
            dependencies,
        }
    }
}

/// A Directed Acyclic Graph (DAG) for task dependencies.
#[derive(Debug, Clone)]
pub struct Dag<T> {
    /// All nodes in the graph, keyed by name.
    nodes: HashMap<String, DagNode<T>>,
}

impl<T> Dag<T> {
    /// Create a new empty DAG.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Add a node to the DAG.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    ///
    /// let mut dag = Dag::new();
    /// dag.add_node("task1", DagNode::new("value1", vec![]));
    /// dag.add_node("task2", DagNode::new("value2", vec!["task1".to_string()]));
    /// ```
    pub fn add_node(&mut self, name: impl Into<String>, node: DagNode<T>) {
        self.nodes.insert(name.into(), node);
    }

    /// Perform topological sort on the DAG.
    ///
    /// Returns the nodes in an order where all dependencies come before
    /// the nodes that depend on them.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut dag = Dag::new();
    /// dag.add_node("task1", DagNode::new(1, vec![]));
    /// dag.add_node("task2", DagNode::new(2, vec!["task1".to_string()]));
    /// dag.add_node("task3", DagNode::new(3, vec!["task1".to_string()]));
    /// dag.add_node("task4", DagNode::new(4, vec!["task2".to_string(), "task3".to_string()]));
    ///
    /// let sorted = dag.topological_sort()?;
    /// let names: Vec<_> = sorted.iter().map(|(name, _)| name.clone()).collect();
    ///
    /// // task1 must come first (no dependencies)
    /// assert_eq!(names[0], "task1");
    ///
    /// // task4 must come last (depends on task2 and task3)
    /// assert_eq!(names[3], "task4");
    /// # Ok(())
    /// # }
    /// ```
    pub fn topological_sort(&self) -> Result<Vec<(String, &T)>, DagError> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize in-degrees and adjacency list
        for name in self.nodes.keys() {
            in_degree.insert(name.clone(), 0);
            adj_list.insert(name.clone(), Vec::new());
        }

        // Build the graph
        for (name, node) in &self.nodes {
            for dep in &node.dependencies {
                // Check that dependency exists
                if !self.nodes.contains_key(dep) {
                    return Err(DagError::UnknownDependency {
                        task: name.clone(),
                        dependency: dep.clone(),
                    });
                }

                // Add edge from dependency to this node
                adj_list.entry(dep.clone()).or_default().push(name.clone());
                *in_degree.entry(name.clone()).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(name) = queue.pop() {
            if let Some(node) = self.nodes.get(&name) {
                result.push((name.clone(), &node.value));
            }

            // Reduce in-degree for all neighbors
            if let Some(neighbors) = adj_list.get(&name) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != self.nodes.len() {
            return Err(DagError::Cycle(
                "Cycle detected in task dependencies".to_string(),
            ));
        }

        Ok(result)
    }

    /// Detect cycles in the DAG.
    ///
    /// Returns `Ok(())` if no cycles are detected, or an error if a cycle is found.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut dag = Dag::new();
    /// dag.add_node("task1", DagNode::new(1, vec!["task2".to_string()]));
    /// dag.add_node("task2", DagNode::new(2, vec!["task1".to_string()]));
    ///
    /// assert!(dag.detect_cycles().is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn detect_cycles(&self) -> Result<(), DagError> {
        self.topological_sort()?;
        Ok(())
    }

    /// Get all nodes that have no dependencies (can be executed first).
    pub fn roots(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.dependencies.is_empty())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get all nodes that are not depended on by any other node (can be executed last).
    pub fn leaves(&self) -> Vec<String> {
        let mut depended_on: HashSet<String> = HashSet::new();

        for node in self.nodes.values() {
            for dep in &node.dependencies {
                depended_on.insert(dep.clone());
            }
        }

        self.nodes
            .keys()
            .filter(|name| !depended_on.contains(*name))
            .cloned()
            .collect()
    }

    /// Get the number of nodes in the DAG.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the DAG is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl<T> Default for Dag<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_sort_simple() {
        let mut dag = Dag::new();
        dag.add_node("a", DagNode::new(1, vec![]));
        dag.add_node("b", DagNode::new(2, vec!["a".to_string()]));
        dag.add_node("c", DagNode::new(3, vec!["b".to_string()]));

        let sorted = dag.topological_sort().unwrap();
        let names: Vec<_> = sorted.iter().map(|(name, _)| name.clone()).collect();

        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_topological_sort_parallel() {
        let mut dag = Dag::new();
        dag.add_node("a", DagNode::new(1, vec![]));
        dag.add_node("b", DagNode::new(2, vec!["a".to_string()]));
        dag.add_node("c", DagNode::new(3, vec!["a".to_string()]));
        dag.add_node("d", DagNode::new(4, vec!["b".to_string(), "c".to_string()]));

        let sorted = dag.topological_sort().unwrap();
        let names: Vec<_> = sorted.iter().map(|(name, _)| name.clone()).collect();

        // a must be first
        assert_eq!(names[0], "a");
        // d must be last
        assert_eq!(names[3], "d");
        // b and c can be in any order
        assert!(names[1..3].iter().any(|s| s == "b"));
        assert!(names[1..3].iter().any(|s| s == "c"));
    }

    #[test]
    fn test_detect_cycles() {
        let mut dag = Dag::new();
        dag.add_node("a", DagNode::new(1, vec!["b".to_string()]));
        dag.add_node("b", DagNode::new(2, vec!["a".to_string()]));

        assert!(matches!(
            dag.detect_cycles(),
            Err(DagError::Cycle(_))
        ));
    }

    #[test]
    fn test_unknown_dependency() {
        let mut dag = Dag::new();
        dag.add_node("a", DagNode::new(1, vec!["nonexistent".to_string()]));

        assert!(matches!(
            dag.topological_sort(),
            Err(DagError::UnknownDependency { .. })
        ));
    }

    #[test]
    fn test_roots() {
        let mut dag = Dag::new();
        dag.add_node("a", DagNode::new(1, vec![]));
        dag.add_node("b", DagNode::new(2, vec!["a".to_string()]));
        dag.add_node("c", DagNode::new(3, vec![]));

        let roots = dag.roots();
        assert_eq!(roots.len(), 2);
        assert!(roots.iter().any(|s| s == "a"));
        assert!(roots.iter().any(|s| s == "c"));
    }

    #[test]
    fn test_leaves() {
        let mut dag = Dag::new();
        dag.add_node("a", DagNode::new(1, vec![]));
        dag.add_node("b", DagNode::new(2, vec!["a".to_string()]));
        dag.add_node("c", DagNode::new(3, vec!["a".to_string()]));

        let leaves = dag.leaves();
        assert_eq!(leaves.len(), 2);
        assert!(leaves.iter().any(|s| s == "b"));
        assert!(leaves.iter().any(|s| s == "c"));
    }
}
