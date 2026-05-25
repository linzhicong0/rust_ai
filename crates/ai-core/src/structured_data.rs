// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Structured data support for inputs/outputs (JSON, tables, CSV).
//!
//! This module provides parsing, schema inference, and format conversion
//! for structured data types like CSV, JSON, and tabular data.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Supported structured data formats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFormat {
    /// JSON format.
    Json,
    /// CSV (comma-separated values).
    Csv,
    /// TSV (tab-separated values).
    Tsv,
    /// Generic tabular data (list of rows with headers).
    Table,
}

/// A row of data in tabular format.
pub type DataRow = HashMap<String, Value>;

/// Inferred type of a column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    /// String type.
    String,
    /// Integer type.
    Integer,
    /// Float type.
    Float,
    /// Boolean type.
    Boolean,
    /// Null/empty.
    Null,
    /// Mixed types detected.
    Mixed,
}

/// Schema for a structured data set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Column names in order.
    pub columns: Vec<String>,
    /// Inferred types for each column.
    pub types: HashMap<String, ColumnType>,
    /// Number of rows in the data.
    pub row_count: usize,
}

/// A structured data set with schema and rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredData {
    /// The inferred schema.
    pub schema: DataSchema,
    /// The data rows.
    pub rows: Vec<DataRow>,
    /// The source format.
    pub format: DataFormat,
}

/// Errors that can occur during structured data operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredDataError {
    /// Failed to parse the input.
    ParseError(String),
    /// Invalid format specified.
    InvalidFormat(String),
    /// Schema validation failed.
    SchemaError(String),
}

impl std::fmt::Display for StructuredDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            Self::SchemaError(msg) => write!(f, "Schema error: {}", msg),
        }
    }
}

impl std::error::Error for StructuredDataError {}

/// Parse CSV data into structured data.
///
/// Expects the first row to be headers.
pub fn parse_csv(input: &str) -> Result<StructuredData, StructuredDataError> {
    parse_delimited(input, ',', DataFormat::Csv)
}

/// Parse TSV data into structured data.
///
/// Expects the first row to be headers.
pub fn parse_tsv(input: &str) -> Result<StructuredData, StructuredDataError> {
    parse_delimited(input, '\t', DataFormat::Tsv)
}

/// Parse delimited data (CSV, TSV, etc.).
fn parse_delimited(
    input: &str,
    delimiter: char,
    format: DataFormat,
) -> Result<StructuredData, StructuredDataError> {
    let lines: Vec<&str> = input.lines().filter(|l| !l.trim().is_empty()).collect();

    if lines.is_empty() {
        return Err(StructuredDataError::ParseError("Empty input".to_string()));
    }

    let headers: Vec<String> = lines[0]
        .split(delimiter)
        .map(|h| h.trim().to_string())
        .collect();

    let mut rows = Vec::new();
    for line in &lines[1..] {
        let values: Vec<&str> = line.split(delimiter).collect();
        let mut row = DataRow::new();

        for (i, header) in headers.iter().enumerate() {
            let value = values.get(i).map(|v| v.trim()).unwrap_or("");
            row.insert(header.clone(), infer_value(value));
        }

        rows.push(row);
    }

    let schema = infer_schema(&headers, &rows);

    Ok(StructuredData {
        schema,
        rows,
        format,
    })
}

/// Parse JSON array data into structured data.
///
/// Input should be a JSON array of objects.
pub fn parse_json(input: &str) -> Result<StructuredData, StructuredDataError> {
    let value: Value =
        serde_json::from_str(input).map_err(|e| StructuredDataError::ParseError(e.to_string()))?;

    match value {
        Value::Array(arr) => {
            let mut columns = Vec::new();
            let mut rows = Vec::new();

            // Collect all column names from all objects
            for item in &arr {
                if let Value::Object(obj) = item {
                    for key in obj.keys() {
                        if !columns.contains(key) {
                            columns.push(key.clone());
                        }
                    }
                }
            }

            // Build rows
            for item in &arr {
                if let Value::Object(obj) = item {
                    let mut row = DataRow::new();
                    for col in &columns {
                        let val = obj.get(col).cloned().unwrap_or(Value::Null);
                        row.insert(col.clone(), val);
                    }
                    rows.push(row);
                }
            }

            let schema = infer_schema(&columns, &rows);

            Ok(StructuredData {
                schema,
                rows,
                format: DataFormat::Json,
            })
        }
        _ => Err(StructuredDataError::ParseError(
            "Expected a JSON array of objects".to_string(),
        )),
    }
}

/// Infer the type of a string value.
fn infer_value(s: &str) -> Value {
    if s.is_empty() {
        return Value::Null;
    }

    // Try boolean
    if s.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if s.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }

    // Try integer
    if let Ok(n) = s.parse::<i64>() {
        return Value::Number(serde_json::Number::from(n));
    }

    // Try float
    if let Ok(n) = s.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(n) {
            return Value::Number(num);
        }
    }

    Value::String(s.to_string())
}

/// Infer the column type from a Value.
fn infer_column_type(value: &Value) -> ColumnType {
    match value {
        Value::Null => ColumnType::Null,
        Value::Bool(_) => ColumnType::Boolean,
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                ColumnType::Integer
            } else {
                ColumnType::Float
            }
        }
        Value::String(_) => ColumnType::String,
        _ => ColumnType::String,
    }
}

/// Merge two column types (for schema inference across rows).
fn merge_types(a: &ColumnType, b: &ColumnType) -> ColumnType {
    if a == b {
        return a.clone();
    }
    if *a == ColumnType::Null {
        return b.clone();
    }
    if *b == ColumnType::Null {
        return a.clone();
    }
    // Integer + Float → Float
    if (*a == ColumnType::Integer && *b == ColumnType::Float)
        || (*a == ColumnType::Float && *b == ColumnType::Integer)
    {
        return ColumnType::Float;
    }
    ColumnType::Mixed
}

/// Infer schema from headers and rows.
fn infer_schema(columns: &[String], rows: &[DataRow]) -> DataSchema {
    let mut types: HashMap<String, ColumnType> = HashMap::new();

    for col in columns {
        let mut col_type = ColumnType::Null;
        for row in rows {
            if let Some(val) = row.get(col) {
                let val_type = infer_column_type(val);
                col_type = merge_types(&col_type, &val_type);
            }
        }
        types.insert(col.clone(), col_type);
    }

    DataSchema {
        columns: columns.to_vec(),
        types,
        row_count: rows.len(),
    }
}

/// Convert structured data to CSV format.
pub fn to_csv(data: &StructuredData) -> String {
    let mut output = String::new();

    // Header row
    output.push_str(&data.schema.columns.join(","));
    output.push('\n');

    // Data rows
    for row in &data.rows {
        let values: Vec<String> = data
            .schema
            .columns
            .iter()
            .map(|col| {
                row.get(col)
                    .map(|v| value_to_csv_cell(v))
                    .unwrap_or_default()
            })
            .collect();
        output.push_str(&values.join(","));
        output.push('\n');
    }

    output
}

/// Convert structured data to JSON array format.
pub fn to_json(data: &StructuredData) -> String {
    let arr: Vec<Value> = data
        .rows
        .iter()
        .map(|row| Value::Object(row.iter().map(|(k, v)| (k.clone(), v.clone())).collect()))
        .collect();

    serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".to_string())
}

/// Convert a Value to a CSV cell string.
fn value_to_csv_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if s.contains(',') || s.contains('"') || s.contains('\n') {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.clone()
            }
        }
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

/// Convert structured data between formats.
pub fn convert(
    data: &StructuredData,
    target_format: DataFormat,
) -> Result<String, StructuredDataError> {
    match target_format {
        DataFormat::Csv => Ok(to_csv(data)),
        DataFormat::Json => Ok(to_json(data)),
        DataFormat::Tsv => {
            let mut output = String::new();
            output.push_str(&data.schema.columns.join("\t"));
            output.push('\n');
            for row in &data.rows {
                let values: Vec<String> = data
                    .schema
                    .columns
                    .iter()
                    .map(|col| {
                        row.get(col)
                            .map(|v| match v {
                                Value::Null => String::new(),
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                output.push_str(&values.join("\t"));
                output.push('\n');
            }
            Ok(output)
        }
        DataFormat::Table => Ok(to_json(data)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_csv_basic() {
        let csv = "name,age,active\nAlice,30,true\nBob,25,false\n";
        let data = parse_csv(csv).unwrap();

        assert_eq!(data.schema.columns, vec!["name", "age", "active"]);
        assert_eq!(data.rows.len(), 2);
        assert_eq!(data.format, DataFormat::Csv);

        assert_eq!(
            data.rows[0].get("name"),
            Some(&Value::String("Alice".to_string()))
        );
        assert_eq!(data.rows[0].get("age"), Some(&Value::Number(30.into())));
        assert_eq!(data.rows[0].get("active"), Some(&Value::Bool(true)));
    }

    #[test]
    fn test_parse_csv_empty_input() {
        let result = parse_csv("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_tsv() {
        let tsv = "id\tname\n1\tAlice\n2\tBob\n";
        let data = parse_tsv(tsv).unwrap();

        assert_eq!(data.schema.columns, vec!["id", "name"]);
        assert_eq!(data.rows.len(), 2);
        assert_eq!(data.format, DataFormat::Tsv);
    }

    #[test]
    fn test_parse_json_array() {
        let json = r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#;
        let data = parse_json(json).unwrap();

        assert_eq!(data.rows.len(), 2);
        assert_eq!(data.format, DataFormat::Json);
        assert!(data.schema.columns.contains(&"name".to_string()));
        assert!(data.schema.columns.contains(&"age".to_string()));
    }

    #[test]
    fn test_parse_json_not_array() {
        let json = r#"{"name": "Alice"}"#;
        let result = parse_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_inference_types() {
        let csv = "int_col,float_col,bool_col,str_col\n1,1.5,true,hello\n2,2.5,false,world\n";
        let data = parse_csv(csv).unwrap();

        assert_eq!(data.schema.types.get("int_col"), Some(&ColumnType::Integer));
        assert_eq!(data.schema.types.get("float_col"), Some(&ColumnType::Float));
        assert_eq!(
            data.schema.types.get("bool_col"),
            Some(&ColumnType::Boolean)
        );
        assert_eq!(data.schema.types.get("str_col"), Some(&ColumnType::String));
    }

    #[test]
    fn test_schema_inference_null_values() {
        let csv = "name,value\nAlice,\nBob,42\n";
        let data = parse_csv(csv).unwrap();

        // value column: Null + Integer → Integer
        assert_eq!(data.schema.types.get("value"), Some(&ColumnType::Integer));
    }

    #[test]
    fn test_schema_inference_mixed_types() {
        let csv = "col\n42\nhello\n";
        let data = parse_csv(csv).unwrap();

        // Integer + String → Mixed
        assert_eq!(data.schema.types.get("col"), Some(&ColumnType::Mixed));
    }

    #[test]
    fn test_to_csv() {
        let csv_input = "name,age\nAlice,30\nBob,25\n";
        let data = parse_csv(csv_input).unwrap();
        let csv_output = to_csv(&data);

        assert!(csv_output.contains("name,age"));
        assert!(csv_output.contains("Alice,30"));
        assert!(csv_output.contains("Bob,25"));
    }

    #[test]
    fn test_to_json() {
        let csv_input = "name,age\nAlice,30\n";
        let data = parse_csv(csv_input).unwrap();
        let json_output = to_json(&data);

        let parsed: Vec<Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["name"], Value::String("Alice".to_string()));
        assert_eq!(parsed[0]["age"], json!(30));
    }

    #[test]
    fn test_convert_csv_to_json() {
        let csv = "x,y\n1,2\n3,4\n";
        let data = parse_csv(csv).unwrap();
        let json_str = convert(&data, DataFormat::Json).unwrap();

        let parsed: Vec<Value> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn test_convert_json_to_csv() {
        let json = r#"[{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]"#;
        let data = parse_json(json).unwrap();
        let csv_str = convert(&data, DataFormat::Csv).unwrap();

        assert!(csv_str.contains("a"));
        assert!(csv_str.contains("b"));
        assert!(csv_str.contains("hello"));
        assert!(csv_str.contains("world"));
    }

    #[test]
    fn test_convert_to_tsv() {
        let csv = "name,age\nAlice,30\n";
        let data = parse_csv(csv).unwrap();
        let tsv = convert(&data, DataFormat::Tsv).unwrap();

        assert!(tsv.contains("name\tage"));
        assert!(tsv.contains("Alice\t30"));
    }

    #[test]
    fn test_csv_with_special_characters() {
        let csv = "name,desc\nAlice,\"hello, world\"\n";
        // Note: our simple parser doesn't handle quoted CSV fields
        // but let's test what happens
        let data = parse_csv(csv).unwrap();
        assert_eq!(data.rows.len(), 1);
    }

    #[test]
    fn test_data_schema_row_count() {
        let csv = "a\n1\n2\n3\n";
        let data = parse_csv(csv).unwrap();
        assert_eq!(data.schema.row_count, 3);
    }

    #[test]
    fn test_infer_value_types() {
        assert_eq!(infer_value(""), Value::Null);
        assert_eq!(infer_value("true"), Value::Bool(true));
        assert_eq!(infer_value("false"), Value::Bool(false));
        assert_eq!(infer_value("42"), Value::Number(42.into()));
        assert_eq!(infer_value("hello"), Value::String("hello".to_string()));
    }

    #[test]
    fn test_infer_value_float() {
        let val = infer_value("3.14");
        assert!(val.is_number());
        assert_eq!(val.as_f64(), Some(3.14));
    }

    #[test]
    fn test_column_type_merge() {
        assert_eq!(
            merge_types(&ColumnType::Null, &ColumnType::Integer),
            ColumnType::Integer
        );
        assert_eq!(
            merge_types(&ColumnType::Integer, &ColumnType::Null),
            ColumnType::Integer
        );
        assert_eq!(
            merge_types(&ColumnType::Integer, &ColumnType::Integer),
            ColumnType::Integer
        );
        assert_eq!(
            merge_types(&ColumnType::Integer, &ColumnType::Float),
            ColumnType::Float
        );
        assert_eq!(
            merge_types(&ColumnType::Integer, &ColumnType::String),
            ColumnType::Mixed
        );
    }

    #[test]
    fn test_structured_data_error_display() {
        let err = StructuredDataError::ParseError("bad input".to_string());
        assert_eq!(err.to_string(), "Parse error: bad input");

        let err = StructuredDataError::InvalidFormat("unknown".to_string());
        assert_eq!(err.to_string(), "Invalid format: unknown");
    }

    #[test]
    fn test_data_format_serialization() {
        let format = DataFormat::Csv;
        let json = serde_json::to_string(&format).unwrap();
        let deserialized: DataFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, DataFormat::Csv);
    }
}
