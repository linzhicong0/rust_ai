// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Data Sources (REQ-17.1)
//!
//! Provides the `DataSource` trait for integrating with common data sources:
//! databases (SQL, NoSQL), file systems, APIs, and message queues.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Errors that can occur during data source operations.
#[derive(Debug, thiserror::Error)]
pub enum DataSourceError {
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Read error: {0}")]
    Read(String),
    #[error("Write error: {0}")]
    Write(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Query error: {0}")]
    Query(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

/// A query to execute against a data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuery {
    /// The collection/table/path to query.
    pub source: String,
    /// Optional filter conditions.
    pub filter: Option<serde_json::Value>,
    /// Maximum number of results.
    pub limit: Option<usize>,
    /// Offset for pagination.
    pub offset: Option<usize>,
}

impl DataQuery {
    /// Create a new query targeting the given source.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            filter: None,
            limit: None,
            offset: None,
        }
    }

    /// Set a filter condition.
    pub fn with_filter(mut self, filter: serde_json::Value) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a limit on results.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set an offset for pagination.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// A record of data from a data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    /// Unique identifier for this record.
    pub id: String,
    /// The data payload.
    pub data: serde_json::Value,
    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

/// The type of data source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSourceType {
    /// SQL databases (PostgreSQL, MySQL, SQLite).
    Sql,
    /// NoSQL databases (MongoDB, Redis).
    NoSql,
    /// File systems (local, S3, GCS).
    File,
    /// HTTP APIs.
    Api,
    /// Message queues.
    MessageQueue,
}

/// Trait for reading from and writing to data sources.
#[async_trait]
pub trait DataSource: Send + Sync {
    /// Get the type of this data source.
    fn source_type(&self) -> DataSourceType;

    /// Read records from the data source matching the given query.
    async fn read(&self, query: DataQuery) -> Result<Vec<DataRecord>, DataSourceError>;

    /// Write a record to the data source.
    async fn write(&self, collection: &str, record: DataRecord) -> Result<String, DataSourceError>;

    /// Delete a record by ID.
    async fn delete(&self, collection: &str, id: &str) -> Result<(), DataSourceError>;

    /// Check if the data source connection is healthy.
    async fn health_check(&self) -> Result<bool, DataSourceError>;
}

/// In-memory data source implementation for testing.
pub struct InMemoryDataSource {
    data: std::sync::Arc<tokio::sync::Mutex<HashMap<String, Vec<DataRecord>>>>,
}

impl InMemoryDataSource {
    /// Create a new in-memory data source.
    pub fn new() -> Self {
        Self {
            data: std::sync::Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryDataSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DataSource for InMemoryDataSource {
    fn source_type(&self) -> DataSourceType {
        DataSourceType::NoSql
    }

    async fn read(&self, query: DataQuery) -> Result<Vec<DataRecord>, DataSourceError> {
        let data = self.data.lock().await;
        let records = data.get(&query.source).cloned().unwrap_or_default();

        let mut results = records;

        // Apply offset
        if let Some(offset) = query.offset {
            if offset < results.len() {
                results = results[offset..].to_vec();
            } else {
                results = Vec::new();
            }
        }

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn write(&self, collection: &str, record: DataRecord) -> Result<String, DataSourceError> {
        let id = record.id.clone();
        let mut data = self.data.lock().await;
        let records = data.entry(collection.to_string()).or_default();
        records.push(record);
        Ok(id)
    }

    async fn delete(&self, collection: &str, id: &str) -> Result<(), DataSourceError> {
        let mut data = self.data.lock().await;
        if let Some(records) = data.get_mut(collection) {
            let len_before = records.len();
            records.retain(|r| r.id != id);
            if records.len() == len_before {
                return Err(DataSourceError::NotFound(format!(
                    "Record {id} not found in {collection}"
                )));
            }
        } else {
            return Err(DataSourceError::NotFound(format!(
                "Collection {collection} not found"
            )));
        }
        Ok(())
    }

    async fn health_check(&self) -> Result<bool, DataSourceError> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_data_query_creation() {
        let query = DataQuery::new("users");
        assert_eq!(query.source, "users");
        assert!(query.filter.is_none());
        assert!(query.limit.is_none());
        assert!(query.offset.is_none());
    }

    #[tokio::test]
    async fn test_data_query_with_options() {
        let query = DataQuery::new("users")
            .with_filter(serde_json::json!({"active": true}))
            .with_limit(10)
            .with_offset(20);

        assert_eq!(query.source, "users");
        assert_eq!(query.filter.unwrap(), serde_json::json!({"active": true}));
        assert_eq!(query.limit.unwrap(), 10);
        assert_eq!(query.offset.unwrap(), 20);
    }

    #[tokio::test]
    async fn test_in_memory_data_source_write_and_read() {
        let ds = InMemoryDataSource::new();

        let record = DataRecord {
            id: "user-1".to_string(),
            data: serde_json::json!({"name": "Alice", "age": 30}),
            metadata: HashMap::new(),
        };

        let id = ds.write("users", record).await.unwrap();
        assert_eq!(id, "user-1");

        let results = ds.read(DataQuery::new("users")).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "user-1");
        assert_eq!(results[0].data["name"], "Alice");
    }

    #[tokio::test]
    async fn test_in_memory_data_source_pagination() {
        let ds = InMemoryDataSource::new();

        for i in 0..10 {
            let record = DataRecord {
                id: format!("item-{i}"),
                data: serde_json::json!({"index": i}),
                metadata: HashMap::new(),
            };
            ds.write("items", record).await.unwrap();
        }

        // Read with limit
        let results = ds
            .read(DataQuery::new("items").with_limit(3))
            .await
            .unwrap();
        assert_eq!(results.len(), 3);

        // Read with offset
        let results = ds
            .read(DataQuery::new("items").with_offset(7))
            .await
            .unwrap();
        assert_eq!(results.len(), 3);

        // Read with limit and offset
        let results = ds
            .read(DataQuery::new("items").with_offset(2).with_limit(3))
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "item-2");
    }

    #[tokio::test]
    async fn test_in_memory_data_source_delete() {
        let ds = InMemoryDataSource::new();

        let record = DataRecord {
            id: "user-1".to_string(),
            data: serde_json::json!({"name": "Alice"}),
            metadata: HashMap::new(),
        };
        ds.write("users", record).await.unwrap();

        ds.delete("users", "user-1").await.unwrap();

        let results = ds.read(DataQuery::new("users")).await.unwrap();
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_in_memory_data_source_delete_not_found() {
        let ds = InMemoryDataSource::new();

        let result = ds.delete("users", "nonexistent").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DataSourceError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_in_memory_data_source_health_check() {
        let ds = InMemoryDataSource::new();
        assert!(ds.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_in_memory_data_source_type() {
        let ds = InMemoryDataSource::new();
        assert_eq!(ds.source_type(), DataSourceType::NoSql);
    }

    #[tokio::test]
    async fn test_in_memory_data_source_read_empty_collection() {
        let ds = InMemoryDataSource::new();
        let results = ds.read(DataQuery::new("nonexistent")).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_in_memory_data_source_multiple_collections() {
        let ds = InMemoryDataSource::new();

        let user = DataRecord {
            id: "user-1".to_string(),
            data: serde_json::json!({"name": "Alice"}),
            metadata: HashMap::new(),
        };
        let order = DataRecord {
            id: "order-1".to_string(),
            data: serde_json::json!({"total": 99.99}),
            metadata: HashMap::new(),
        };

        ds.write("users", user).await.unwrap();
        ds.write("orders", order).await.unwrap();

        let users = ds.read(DataQuery::new("users")).await.unwrap();
        let orders = ds.read(DataQuery::new("orders")).await.unwrap();

        assert_eq!(users.len(), 1);
        assert_eq!(orders.len(), 1);
        assert_eq!(users[0].data["name"], "Alice");
        assert_eq!(orders[0].data["total"], 99.99);
    }
}
