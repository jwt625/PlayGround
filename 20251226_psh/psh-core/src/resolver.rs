//! Snippet resolver and template renderer
//!
//! Resolves directives against snippet collection and renders templates with Tera.

use crate::parser::{Directive, Operation, Segment};
use crate::snippet::{Snippet, SnippetCollection};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during resolution and rendering
#[derive(Debug, Error)]
pub enum ResolverError {
    /// Template rendering failed
    #[error("Template rendering failed: {0}")]
    RenderError(String),
    
    /// Snippet not found
    #[error("Snippet not found for namespace: {0}")]
    SnippetNotFound(String),
}

/// Warning about unknown elements in directives
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Warning {
    /// Unknown namespace
    UnknownNamespace(String),
    
    /// Unknown operation
    UnknownOperation {
        /// The namespace
        namespace: String,
        /// The unknown operation
        op: String,
    },
    
    /// Unknown key
    UnknownKey {
        /// The namespace
        namespace: String,
        /// The unknown key
        key: String,
    },
}

/// Result of resolving and rendering a directive
#[derive(Debug, Clone)]
pub struct ResolvedDirective {
    /// The original directive
    pub directive: Directive,
    
    /// The rendered text
    pub rendered: String,
    
    /// Warnings about unknown elements
    pub warnings: Vec<Warning>,
}

/// Resolver for expanding directives using snippets
pub struct Resolver {
    /// Snippet collection
    snippets: SnippetCollection,
}

impl Resolver {
    /// Create a new resolver with the given snippet collection
    #[must_use]
    pub const fn new(snippets: SnippetCollection) -> Self {
        Self { snippets }
    }
    
    /// Resolve and render a directive
    ///
    /// # Errors
    ///
    /// Returns `ResolverError` if rendering fails
    pub fn resolve(&self, directive: &Directive) -> Result<ResolvedDirective, ResolverError> {
        let mut rendered_parts = Vec::new();
        let mut warnings = Vec::new();
        
        for segment in &directive.segments {
            let (text, mut segment_warnings) = self.resolve_segment(segment)?;
            rendered_parts.push(text);
            warnings.append(&mut segment_warnings);
        }
        
        Ok(ResolvedDirective {
            directive: directive.clone(),
            rendered: rendered_parts.join("\n\n"),
            warnings,
        })
    }
    
    /// Resolve a single segment
    fn resolve_segment(&self, segment: &Segment) -> Result<(String, Vec<Warning>), ResolverError> {
        let mut warnings = Vec::new();

        // Get the snippet
        let Some(snippet) = self.snippets.get(&segment.namespace) else {
            warnings.push(Warning::UnknownNamespace(segment.namespace.clone()));
            return Ok((String::new(), warnings));
        };

        // Build context by applying operations (global first, then namespace-specific)
        let (context, mut op_warnings) = self.build_context(snippet, &segment.operations);
        warnings.append(&mut op_warnings);

        // Render template
        let rendered = Self::render_template(&snippet.template, &context)?;

        Ok((rendered, warnings))
    }

    /// Build template context from operations
    /// 1. Apply global base op (if it exists)
    /// 2. Apply snippet-specific base op (if it exists, overrides global base)
    /// 3. Apply user-specified ops (global first, then snippet-specific)
    /// 4. Apply key-value pairs (override everything)
    ///
    /// Later ops override earlier ones
    fn build_context(
        &self,
        snippet: &Snippet,
        operations: &[Operation],
    ) -> (HashMap<String, String>, Vec<Warning>) {
        let mut context = HashMap::new();
        let mut warnings = Vec::new();

        // Step 1: Apply global base op if it exists
        if let Some(variables) = self.snippets.global_ops().get("base") {
            for (key, value) in variables {
                context.insert(key.clone(), value.clone());
            }
        }

        // Step 2: Apply snippet-specific base op if it exists (overrides global base)
        if let Some(variables) = snippet.ops.get("base") {
            for (key, value) in variables {
                context.insert(key.clone(), value.clone());
            }
        }

        // Step 3: Apply user-specified operations
        for operation in operations {
            match operation {
                Operation::Op(op_code) => {
                    // Handle namespace-scoped ops (e.g., "d.ne", "sum.l1", or "git.cm.l1")
                    // For hierarchical namespaces like "git.cm", we need to find where the namespace ends
                    // and the op begins. We do this by checking if the op_code starts with the namespace.
                    let actual_op = if op_code.starts_with(&format!("{}.", snippet.namespace)) {
                        // Strip the namespace prefix (e.g., "d.ne" -> "ne", "git.cm.l1" -> "l1")
                        &op_code[snippet.namespace.len() + 1..]
                    } else if op_code.contains('.') {
                        // If it has a dot but doesn't match our namespace, it's an error
                        warnings.push(Warning::UnknownOperation {
                            namespace: snippet.namespace.clone(),
                            op: op_code.clone(),
                        });
                        continue;
                    } else {
                        // No namespace prefix, use as-is
                        op_code.as_str()
                    };

                    // Try to find the op in snippet-specific ops first
                    let mut found = false;
                    if let Some(variables) = snippet.ops.get(actual_op) {
                        // Apply all variables from this snippet-specific op
                        for (key, value) in variables {
                            context.insert(key.clone(), value.clone());
                        }
                        found = true;
                    }

                    // If not found in snippet, try global ops
                    if !found {
                        if let Some(variables) = self.snippets.global_ops().get(actual_op) {
                            // Apply all variables from this global op
                            for (key, value) in variables {
                                context.insert(key.clone(), value.clone());
                            }
                            found = true;
                        }
                    }

                    // If still not found, it's an unknown op
                    if !found {
                        warnings.push(Warning::UnknownOperation {
                            namespace: snippet.namespace.clone(),
                            op: op_code.clone(),
                        });
                    }
                }
                Operation::KeyValue { key, value } => {
                    // Check if key exists in snippet ops or global ops
                    let key_exists = snippet.ops.values().any(|op| op.contains_key(key))
                        || self.snippets.global_ops().values().any(|op| op.contains_key(key));

                    if !key_exists {
                        warnings.push(Warning::UnknownKey {
                            namespace: snippet.namespace.clone(),
                            key: key.clone(),
                        });
                    }

                    // Apply the key-value override (last wins)
                    context.insert(key.clone(), value.clone());
                }
            }
        }

        (context, warnings)
    }

    /// Render a template with the given context
    fn render_template(
        template: &str,
        context: &HashMap<String, String>,
    ) -> Result<String, ResolverError> {
        // Create a Tera context
        let mut tera_context = tera::Context::new();
        for (key, value) in context {
            tera_context.insert(key, value);
        }

        // Render the template using one_off since we don't need template caching
        tera::Tera::one_off(template, &tera_context, false)
            .map_err(|e| ResolverError::RenderError(format!("{e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_directives;

    fn create_test_collection() -> SnippetCollection {
        let toml = r#"
[[snippet]]
id = "doc-style"
namespace = "d"
template = "Provide {{ style }} documentation with {{ emoji }} emoji."

[snippet.ops.base]
style = "detailed"
emoji = "true"

[snippet.ops.ne]
style = "detailed"
emoji = "false"

[snippet.ops.l1]
style = "concise"
emoji = "true"

[snippet.ops.l5]
style = "very detailed"
emoji = "true"

[[snippet]]
id = "code-review"
namespace = "cr"
template = "Review this {{ lang }} code with {{ detail }} detail."

[snippet.ops.base]
lang = "generic"
detail = "medium"
"#;

        let mut collection = SnippetCollection::new();
        collection.load_from_string(toml).unwrap();
        collection
    }

    #[test]
    fn test_resolve_simple_directive() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;d,ne");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Provide detailed documentation with false emoji.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_resolve_with_multiple_ops() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // l5 comes after ne, so l5's emoji=true overrides ne's emoji=false
        let directives = parse_directives(";;d,ne,l5");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Provide very detailed documentation with true emoji.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_resolve_with_key_value() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;cr,base,lang=python,detail=high");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Review this python code with high detail.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_unknown_namespace() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;unknown,op1");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "");
        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownNamespace(_)));
    }

    #[test]
    fn test_unknown_operation() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Provide a valid op first, then an unknown one
        let directives = parse_directives(";;d,ne,unknown_op");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownOperation { .. }));
    }

    #[test]
    fn test_unknown_key() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Provide a valid op first, then an unknown key
        let directives = parse_directives(";;d,ne,unknown_key=value");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownKey { .. }));
    }

    #[test]
    fn test_multiple_segments() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;d,ne;cr,base,lang=rust");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert!(result.rendered.contains("detailed documentation"));
        assert!(result.rendered.contains("Review this rust code"));
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_namespace_scoped_ops() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Test that both syntaxes work: "ne" and "d.ne"
        let directives1 = parse_directives(";;d,ne");
        let result1 = resolver.resolve(&directives1[0]).unwrap();

        let directives2 = parse_directives(";;d,d.ne");
        let result2 = resolver.resolve(&directives2[0]).unwrap();

        // Both should produce the same result
        assert_eq!(result1.rendered, result2.rendered);
        assert_eq!(result1.rendered, "Provide detailed documentation with false emoji.");
        assert!(result1.warnings.is_empty());
        assert!(result2.warnings.is_empty());
    }

    #[test]
    fn test_namespace_scoped_ops_mixed() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Test mixed syntax: some ops with namespace prefix, some without
        let directives = parse_directives(";;d,ne,d.l5");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Provide very detailed documentation with true emoji.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_namespace_scoped_ops_wrong_namespace() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Test that using wrong namespace prefix produces a warning
        // We need to provide at least one valid op so the template can render
        let directives = parse_directives(";;d,ne,cr.base");
        let result = resolver.resolve(&directives[0]).unwrap();

        // Should have one warning for the wrong namespace
        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownOperation { .. }));

        // But the valid op should still work
        assert!(result.rendered.contains("detailed documentation"));
    }
}

