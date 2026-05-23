//! Directed Acyclic Graph (DAG) resolution for task dependencies.
//!
//! This module provides utilities for working with task dependency graphs,
//! including topological sorting and cycle detection.

use crate::step::{Step, StepKind, Task};
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

    /// Get the execution order for tasks based on dependencies.
    ///
    /// Returns a list of task names in the order they should be executed.
    /// Tasks with no dependencies come first, followed by tasks that depend on them.
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
    ///
    /// let order = dag.execution_order()?;
    /// assert_eq!(order, vec!["task1", "task2"]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn execution_order(&self) -> Result<Vec<String>, DagError> {
        let sorted = self.topological_sort()?;
        Ok(sorted.into_iter().map(|(name, _)| name).collect())
    }

    /// Get tasks that can be executed in parallel at each level.
    ///
    /// This groups tasks by their "depth" in the dependency graph, where
    /// all tasks at depth 0 have no dependencies, tasks at depth 1 depend
    /// only on depth 0 tasks, etc.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut dag = Dag::new();
    /// dag.add_node("a", DagNode::new(1, vec![]));
    /// dag.add_node("b", DagNode::new(2, vec![]));
    /// dag.add_node("c", DagNode::new(3, vec!["a".to_string()]));
    /// dag.add_node("d", DagNode::new(4, vec!["a".to_string(), "b".to_string()]));
    ///
    /// let levels = dag.execution_levels()?;
    /// // Level 0: ["a", "b"] (no dependencies)
    /// // Level 1: ["c"] (depends on "a")
    /// // Level 2: ["d"] (depends on "a" and "b")
    /// # Ok(())
    /// # }
    /// ```
    pub fn execution_levels(&self) -> Result<Vec<Vec<String>>, DagError> {
        let order = self.topological_sort()?;
        let mut depth: HashMap<String, usize> = HashMap::new();

        // Calculate depth for each node
        for (name, node) in &self.nodes {
            let max_dep_depth = node
                .dependencies
                .iter()
                .map(|dep| depth.get(dep).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            depth.insert(name.clone(), max_dep_depth + 1);
        }

        // Group by depth
        let max_depth = depth.values().copied().max().unwrap_or(0);
        let mut levels = vec![Vec::new(); max_depth];

        for (name, d) in &depth {
            levels.get_mut(d - 1).unwrap().push(name.clone());
        }

        // Sort levels by original topological order
        for level in &mut levels {
            level.sort_by_key(|name| {
                order.iter().position(|(n, _)| n == name).unwrap_or(usize::MAX)
            });
        }

        levels.retain(|l| !l.is_empty());
        Ok(levels)
    }

    /// Check if a specific task can be executed given a set of completed tasks.
    ///
    /// # Arguments
    ///
    /// * `task_name` - The task to check
    /// * `completed` - Set of task names that have been completed
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    /// use std::collections::HashSet;
    ///
    /// let mut dag = Dag::new();
    /// dag.add_node("task1", DagNode::new(1, vec![]));
    /// dag.add_node("task2", DagNode::new(2, vec!["task1".to_string()]));
    ///
    /// let completed = HashSet::from(["task1".to_string()]);
    /// assert!(dag.can_execute("task2", &completed));
    ///
    /// let completed = HashSet::new();
    /// assert!(!dag.can_execute("task2", &completed));
    /// ```
    pub fn can_execute(&self, task_name: &str, completed: &HashSet<String>) -> bool {
        if let Some(node) = self.nodes.get(task_name) {
            node.dependencies.iter().all(|dep| completed.contains(dep))
        } else {
            false
        }
    }

    /// Get tasks that are ready to execute given a set of completed tasks.
    ///
    /// Returns tasks whose dependencies are all satisfied, excluding already
    /// completed tasks.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    /// use std::collections::HashSet;
    ///
    /// let mut dag = Dag::new();
    /// dag.add_node("task1", DagNode::new(1, vec![]));
    /// dag.add_node("task2", DagNode::new(2, vec!["task1".to_string()]));
    /// dag.add_node("task3", DagNode::new(3, vec![]));
    ///
    /// let completed = HashSet::from(["task1".to_string()]);
    /// let ready = dag.ready_tasks(&completed);
    /// // Returns: ["task2", "task3"] (task3 has no deps, task2's dep is satisfied)
    /// ```
    pub fn ready_tasks(&self, completed: &HashSet<String>) -> Vec<String> {
        self.nodes
            .keys()
            .filter(|name| !completed.contains(*name))
            .filter(|name| self.can_execute(name, completed))
            .cloned()
            .collect()
    }

    /// Get a dependency chain from one task to another.
    ///
    /// Returns a path through the dependency graph from `start` to `end`,
    /// or `None` if no such path exists.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    ///
    /// let mut dag = Dag::new();
    /// dag.add_node("a", DagNode::new(1, vec![]));
    /// dag.add_node("b", DagNode::new(2, vec!["a".to_string()]));
    /// dag.add_node("c", DagNode::new(3, vec!["b".to_string()]));
    ///
    /// let path = dag.dependency_chain("c", "a");
    /// assert_eq!(
    ///     path,
    ///     Some(vec!["c".to_string(), "b".to_string(), "a".to_string()])
    /// );
    /// ```
    pub fn dependency_chain(&self, start: &str, end: &str) -> Option<Vec<String>> {
        if !self.nodes.contains_key(start) || !self.nodes.contains_key(end) {
            return None;
        }

        let mut visited = HashSet::new();
        let mut path = Vec::new();

        self.dfs_path(start, end, &mut path, &mut visited);
        if path.last().map(|node| node.as_str()) == Some(end) {
            Some(path)
        } else {
            None
        }
    }

    /// DFS helper for finding a path.
    fn dfs_path(
        &self,
        current: &str,
        target: &str,
        path: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) -> bool {
        visited.insert(current.to_string());
        path.push(current.to_string());

        if current == target {
            return true;
        }

        if let Some(node) = self.nodes.get(current) {
            for dep in &node.dependencies {
                if !visited.contains(dep) {
                    if self.dfs_path(dep, target, path, visited) {
                        return true;
                    }
                }
            }
        }

        path.pop();
        false
    }

    /// Get all tasks that transitively depend on the given task.
    ///
    /// This is useful for determining which tasks would be affected if
    /// a task fails or changes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ai_pipeline::dag::{Dag, DagNode};
    ///
    /// let mut dag = Dag::new();
    /// dag.add_node("a", DagNode::new(1, vec![]));
    /// dag.add_node("b", DagNode::new(2, vec!["a".to_string()]));
    /// dag.add_node("c", DagNode::new(3, vec!["b".to_string()]));
    ///
    /// let downstream = dag.downstream_tasks("a");
    /// // Returns: ["b", "c"] (both transitively depend on "a")
    /// ```
    pub fn downstream_tasks(&self, task: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        for (name, node) in &self.nodes {
            if name != task && node.dependencies.contains(&task.to_string()) {
                self.collect_downstream(name, &mut result, &mut visited);
            }
        }

        result
    }

    /// Helper to collect all downstream tasks.
    fn collect_downstream(
        &self,
        task: &str,
        result: &mut Vec<String>,
        visited: &mut HashSet<String>,
    ) {
        if visited.contains(task) {
            return;
        }
        visited.insert(task.to_string());
        result.push(task.to_string());

        for (name, node) in &self.nodes {
            if node.dependencies.contains(&task.to_string()) {
                self.collect_downstream(name, result, visited);
            }
        }
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

impl Dag<Task> {
    /// Build a DAG from a slice of steps with their names.
    ///
    /// This extracts Task steps and their dependencies, building a DAG
    /// that can be used for topological sorting.
    pub fn from_steps(steps: &[Step]) -> Result<Self, DagError> {
        let mut dag = Self::new();

        for step in steps {
            if let StepKind::Task(ref task) = &step.kind {
                dag.add_node(&step.name, DagNode::new(task.clone(), task.dependencies.clone()));
            }
        }

        dag.detect_cycles()?;
        Ok(dag)
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
