//! Snippet system for psh
//!
//! Defines the snippet schema, loader, and resolver for expanding directives.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur in the snippet system
#[derive(Debug, Error)]
pub enum SnippetError {
    /// Failed to load snippet file
    #[error("Failed to load snippet file: {0}")]
    LoadError(String),
    
    /// Invalid snippet schema
    #[error("Invalid snippet schema: {0}")]
    InvalidSchema(String),
    
    /// Snippet not found
    #[error("Snippet not found for namespace: {0}")]
    NotFound(String),
}

/// A snippet definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Snippet {
    /// Unique identifier
    pub id: String,

    /// Namespace path (e.g., "d", "py.venv")
    pub namespace: String,

    /// Template text with Tera syntax
    pub template: String,

    /// Operation definitions (op code -> complete variable sets)
    /// Each op defines all variables needed for the template
    #[serde(default)]
    pub ops: HashMap<String, HashMap<String, String>>,

    /// Tags for categorization and search
    #[serde(default)]
    pub tags: Vec<String>,

    /// Human-readable description
    #[serde(default)]
    pub description: String,
}

/// Global operations that apply across all namespaces
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlobalOps {
    /// Operation definitions (op code -> complete variable sets)
    #[serde(default)]
    pub ops: HashMap<String, HashMap<String, String>>,
}

/// Collection of snippets loaded from TOML files
#[derive(Debug, Clone)]
pub struct SnippetCollection {
    /// Snippets indexed by namespace
    snippets: HashMap<String, Snippet>,

    /// Global operations that apply to all namespaces
    global_ops: HashMap<String, HashMap<String, String>>,
}

impl SnippetCollection {
    /// Create a new empty collection
    #[must_use]
    pub fn new() -> Self {
        Self {
            snippets: HashMap::new(),
            global_ops: HashMap::new(),
        }
    }

    /// Load snippets from a TOML file
    ///
    /// # Errors
    ///
    /// Returns `SnippetError::LoadError` if the file cannot be read or parsed
    pub fn load_from_file(&mut self, path: &Path) -> Result<(), SnippetError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| SnippetError::LoadError(format!("Cannot read file: {e}")))?;

        self.load_from_string(&content)
    }

    /// Load snippets from a TOML string
    ///
    /// # Errors
    ///
    /// Returns `SnippetError::InvalidSchema` if the TOML is invalid
    pub fn load_from_string(&mut self, content: &str) -> Result<(), SnippetError> {
        #[derive(Deserialize)]
        struct SnippetFile {
            #[serde(default)]
            snippet: Vec<Snippet>,
            #[serde(default)]
            global: Option<GlobalOps>,
        }

        let file: SnippetFile = toml::from_str(content)
            .map_err(|e| SnippetError::InvalidSchema(format!("Invalid TOML: {e}")))?;

        // Load global ops if present
        if let Some(global) = file.global {
            self.global_ops = global.ops;
        }

        for snippet in file.snippet {
            // Validate snippet
            if snippet.id.is_empty() {
                return Err(SnippetError::InvalidSchema(
                    "Snippet ID cannot be empty".to_string(),
                ));
            }

            if snippet.namespace.is_empty() {
                return Err(SnippetError::InvalidSchema(
                    format!("Snippet '{}' has empty namespace", snippet.id),
                ));
            }

            // Store by namespace
            self.snippets.insert(snippet.namespace.clone(), snippet);
        }

        Ok(())
    }
    
    /// Get a snippet by namespace
    #[must_use]
    pub fn get(&self, namespace: &str) -> Option<&Snippet> {
        self.snippets.get(namespace)
    }

    /// Get global ops
    #[must_use]
    pub fn global_ops(&self) -> &HashMap<String, HashMap<String, String>> {
        &self.global_ops
    }

    /// Get all snippets
    #[must_use]
    pub fn all(&self) -> Vec<&Snippet> {
        self.snippets.values().collect()
    }

    /// Check if a namespace exists
    #[must_use]
    pub fn contains(&self, namespace: &str) -> bool {
        self.snippets.contains_key(namespace)
    }
}

impl Default for SnippetCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_snippet_from_string() {
        let toml = r#"
[[snippet]]
id = "doc-style"
namespace = "d"
template = "Provide {{ style }} documentation"
description = "Documentation style response"

[snippet.ops.detailed]
style = "detailed"

[snippet.ops.concise]
style = "concise"
"#;

        let mut collection = SnippetCollection::new();
        collection.load_from_string(toml).unwrap();

        let snippet = collection.get("d").unwrap();
        assert_eq!(snippet.id, "doc-style");
        assert_eq!(snippet.namespace, "d");
        assert_eq!(snippet.template, "Provide {{ style }} documentation");
        assert_eq!(snippet.ops.get("detailed").unwrap().get("style").unwrap(), "detailed");
    }

    #[test]
    fn test_multiple_snippets() {
        let toml = r#"
[[snippet]]
id = "doc-style"
namespace = "d"
template = "Documentation"

[[snippet]]
id = "summarize"
namespace = "sum"
template = "Summarize"
"#;

        let mut collection = SnippetCollection::new();
        collection.load_from_string(toml).unwrap();

        assert!(collection.contains("d"));
        assert!(collection.contains("sum"));
        assert_eq!(collection.all().len(), 2);
    }

    #[test]
    fn test_invalid_schema() {
        let toml = r#"
[[snippet]]
id = ""
namespace = "d"
template = "Test"
"#;

        let mut collection = SnippetCollection::new();
        let result = collection.load_from_string(toml);

        assert!(result.is_err());
    }

    #[test]
    fn test_snippet_with_ops() {
        let toml = r#"
[[snippet]]
id = "test"
namespace = "t"
template = "Test {{ length }}"

[snippet.ops.l1]
length = "short"

[snippet.ops.l5]
length = "very long"
"#;

        let mut collection = SnippetCollection::new();
        collection.load_from_string(toml).unwrap();

        let snippet = collection.get("t").unwrap();
        assert_eq!(snippet.ops.len(), 2);
        assert_eq!(snippet.ops.get("l1").unwrap().get("length").unwrap(), "short");
        assert_eq!(snippet.ops.get("l5").unwrap().get("length").unwrap(), "very long");
    }
}

