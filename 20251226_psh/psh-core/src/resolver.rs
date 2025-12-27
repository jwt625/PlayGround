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

        // Handle global-only directives (no namespace, just ops)
        if segment.namespace == "_global" {
            return self.resolve_global_segment(segment);
        }

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

    /// Resolve a global-only segment (no namespace, just global ops)
    ///
    /// For global-only directives like ";;ne,l5", we apply the global ops
    /// and render a simple template that outputs the instructions.
    fn resolve_global_segment(&self, segment: &Segment) -> Result<(String, Vec<Warning>), ResolverError> {
        let mut context = HashMap::new();
        let mut warnings = Vec::new();

        // Apply global base op if it exists
        if let Some(variables) = self.snippets.global_ops().get("base") {
            for (key, value) in variables {
                context.insert(key.clone(), value.clone());
            }
        }

        // Apply user-specified global operations
        for operation in &segment.operations {
            match operation {
                Operation::Op(op_code) => {
                    // For global segments, we only look up global ops
                    if let Some(variables) = self.snippets.global_ops().get(op_code.as_str()) {
                        for (key, value) in variables {
                            context.insert(key.clone(), value.clone());
                        }
                    } else {
                        // Unknown global op
                        warnings.push(Warning::UnknownOperation {
                            namespace: "_global".to_string(),
                            op: op_code.clone(),
                        });
                    }
                }
                Operation::KeyValue { key, value } => {
                    // Check if key exists in any global op
                    let key_exists = self.snippets.global_ops().values().any(|op| op.contains_key(key));

                    if !key_exists {
                        warnings.push(Warning::UnknownKey {
                            namespace: "_global".to_string(),
                            key: key.clone(),
                        });
                    }

                    // Apply the key-value override
                    context.insert(key.clone(), value.clone());
                }
            }
        }

        // Render a simple template that outputs the instructions
        // We'll concatenate all non-empty instruction values
        let template = Self::build_global_template(&context);
        let rendered = Self::render_template(&template, &context)?;

        Ok((rendered, warnings))
    }

    /// Build a template for global-only directives
    ///
    /// Creates a template that outputs all the instruction variables in a sensible order
    fn build_global_template(context: &HashMap<String, String>) -> String {
        let mut parts = Vec::new();

        // Order: length, emoji, tone, format, detail, then any others
        let ordered_keys = [
            "length_instruction",
            "emoji_instruction",
            "tone_instruction",
            "format_instruction",
            "detail_instruction",
        ];

        for key in &ordered_keys {
            if let Some(value) = context.get(*key) {
                if !value.is_empty() {
                    parts.push(format!("{{{{ {key} }}}}"));
                }
            }
        }

        // Add any other keys not in the ordered list
        for (key, value) in context {
            if !ordered_keys.contains(&key.as_str()) && !value.is_empty() {
                parts.push(format!("{{{{ {key} }}}}"));
            }
        }

        if parts.is_empty() {
            // No instructions, return empty
            String::new()
        } else {
            // Join with newlines
            parts.join("\n")
        }
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

        let directives = parse_directives(";;d.ne");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Provide detailed documentation with false emoji.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_resolve_with_multiple_ops() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // l5 comes after ne, so l5's emoji=true overrides ne's emoji=false
        let directives = parse_directives(";;d.ne,l5");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Provide very detailed documentation with true emoji.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_resolve_with_key_value() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;cr.base,lang=python,detail=high");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Review this python code with high detail.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_unknown_namespace() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;unknown.op1");
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
        let directives = parse_directives(";;d.ne,unknown_op");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownOperation { .. }));
    }

    #[test]
    fn test_unknown_key() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Provide a valid op first, then an unknown key
        let directives = parse_directives(";;d.ne,unknown_key=value");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownKey { .. }));
    }

    #[test]
    fn test_multiple_segments() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;d.ne;cr.base,lang=rust");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert!(result.rendered.contains("detailed documentation"));
        assert!(result.rendered.contains("Review this rust code"));
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_namespace_scoped_ops() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Test namespace.op syntax
        let directives = parse_directives(";;d.ne");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Provide detailed documentation with false emoji.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_namespace_scoped_ops_mixed() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Test multiple ops: namespace.op1,op2
        let directives = parse_directives(";;d.ne,l5");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert_eq!(result.rendered, "Provide very detailed documentation with true emoji.");
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_namespace_scoped_ops_wrong_namespace() {
        let collection = create_test_collection();
        let resolver = Resolver::new(collection);

        // Test that using unknown op produces a warning
        let directives = parse_directives(";;cr.unknown_op");
        let result = resolver.resolve(&directives[0]).unwrap();

        // Should have one warning for the unknown op
        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownOperation { .. }));

        // But the template should still render with base values
        assert!(result.rendered.contains("Review this generic code"));
    }

    fn create_test_collection_with_global_ops() -> SnippetCollection {
        let toml = r#"
[global]

[global.ops.base]
length_instruction = "Provide a moderate response."
emoji_instruction = "No emoji."
tone_instruction = "Professional tone."

[global.ops.l1]
length_instruction = "Be very concise."

[global.ops.l5]
length_instruction = "Be very detailed."

[global.ops.ne]
emoji_instruction = "No emoji at all."

[global.ops.pro]
tone_instruction = "Strictly professional."

[[snippet]]
id = "doc"
namespace = "d"
template = "Documentation: {{ length_instruction }} {{ emoji_instruction }}"

[snippet.ops.base]
length_instruction = "Moderate docs."
emoji_instruction = "No emoji."
"#;

        let mut collection = SnippetCollection::new();
        collection.load_from_string(toml).unwrap();
        collection
    }

    #[test]
    fn test_global_only_single_op() {
        let collection = create_test_collection_with_global_ops();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;ne");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert!(result.rendered.contains("No emoji at all"));
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_global_only_multiple_ops() {
        let collection = create_test_collection_with_global_ops();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;ne,l5");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert!(result.rendered.contains("Be very detailed"));
        assert!(result.rendered.contains("No emoji at all"));
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_global_only_with_tone() {
        let collection = create_test_collection_with_global_ops();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;l5,pro");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert!(result.rendered.contains("Be very detailed"));
        assert!(result.rendered.contains("Strictly professional"));
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_global_only_unknown_op() {
        let collection = create_test_collection_with_global_ops();
        let resolver = Resolver::new(collection);

        let directives = parse_directives(";;ne,unknown_op");
        let result = resolver.resolve(&directives[0]).unwrap();

        // Should have warning for unknown op
        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownOperation { .. }));

        // But known op should still work
        assert!(result.rendered.contains("No emoji at all"));
    }

    #[test]
    fn test_global_only_with_key_value() {
        let collection = create_test_collection_with_global_ops();
        let resolver = Resolver::new(collection);

        // Note: directives end at whitespace, so we use a single-word value
        let directives = parse_directives(";;ne,length_instruction=CustomLength");
        let result = resolver.resolve(&directives[0]).unwrap();

        assert!(result.rendered.contains("CustomLength"));
        assert!(result.rendered.contains("No emoji at all"));
        assert!(result.warnings.is_empty());
    }
}

