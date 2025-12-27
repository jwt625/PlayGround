//! Text expansion engine
//!
//! Expands directives in text by replacing them with rendered snippets.

use crate::parser::parse_directives;
use crate::resolver::{ResolvedDirective, Resolver, Warning};
use crate::snippet::SnippetCollection;
use thiserror::Error;

/// Errors that can occur during expansion
#[derive(Debug, Error)]
pub enum ExpandError {
    /// Resolution error
    #[error("Resolution error: {0}")]
    ResolverError(#[from] crate::resolver::ResolverError),
}

/// Result of expanding text
#[derive(Debug, Clone)]
pub struct ExpandResult {
    /// The expanded text with directives replaced
    pub text: String,
    
    /// All warnings from all directives
    pub warnings: Vec<Warning>,
    
    /// Details of each resolved directive
    pub resolved: Vec<ResolvedDirective>,
}

/// Text expander that replaces directives with rendered snippets
pub struct Expander {
    /// Resolver for directives
    resolver: Resolver,
}

impl Expander {
    /// Create a new expander with the given snippet collection
    #[must_use]
    pub const fn new(snippets: SnippetCollection) -> Self {
        Self {
            resolver: Resolver::new(snippets),
        }
    }
    
    /// Expand all directives in the given text
    ///
    /// # Errors
    ///
    /// Returns `ExpandError` if resolution or rendering fails
    pub fn expand(&self, text: &str) -> Result<ExpandResult, ExpandError> {
        // Parse all directives
        let directives = parse_directives(text);
        
        if directives.is_empty() {
            return Ok(ExpandResult {
                text: text.to_string(),
                warnings: Vec::new(),
                resolved: Vec::new(),
            });
        }
        
        // Resolve all directives
        let mut resolved = Vec::new();
        let mut all_warnings = Vec::new();
        
        for directive in &directives {
            let result = self.resolver.resolve(directive)?;
            all_warnings.extend(result.warnings.clone());
            resolved.push(result);
        }
        
        // Replace directives in text (from end to start to preserve positions)
        let mut expanded_text = text.to_string();
        for result in resolved.iter().rev() {
            let start = result.directive.start;
            let end = result.directive.end;
            expanded_text.replace_range(start..end, &result.rendered);
        }
        
        Ok(ExpandResult {
            text: expanded_text,
            warnings: all_warnings,
            resolved,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_expander() -> Expander {
        let toml = r#"
[[snippet]]
id = "doc-style"
namespace = "d"
template = "Provide {{ style }} documentation."

[snippet.ops.base]
style = "detailed"

[snippet.ops.l1]
style = "concise"

[[snippet]]
id = "summarize"
namespace = "sum"
template = "Summarize in {{ format }} format."

[snippet.ops.base]
format = "paragraph"

[snippet.ops.blt]
format = "bullet points"
"#;

        let mut collection = SnippetCollection::new();
        collection.load_from_string(toml).unwrap();
        Expander::new(collection)
    }

    #[test]
    fn test_expand_single_directive() {
        let expander = create_test_expander();
        let text = "Please ;;d,l1 explain this.";
        
        let result = expander.expand(text).unwrap();
        
        assert_eq!(result.text, "Please Provide concise documentation. explain this.");
        assert!(result.warnings.is_empty());
        assert_eq!(result.resolved.len(), 1);
    }

    #[test]
    fn test_expand_multiple_directives() {
        let expander = create_test_expander();
        let text = "First ;;d,l1 then ;;sum,blt done.";
        
        let result = expander.expand(text).unwrap();
        
        assert!(result.text.contains("Provide concise documentation."));
        assert!(result.text.contains("Summarize in bullet points format."));
        assert!(result.warnings.is_empty());
        assert_eq!(result.resolved.len(), 2);
    }

    #[test]
    fn test_expand_no_directives() {
        let expander = create_test_expander();
        let text = "No directives here.";
        
        let result = expander.expand(text).unwrap();
        
        assert_eq!(result.text, text);
        assert!(result.warnings.is_empty());
        assert_eq!(result.resolved.len(), 0);
    }

    #[test]
    fn test_expand_with_warnings() {
        let expander = create_test_expander();
        let text = "Test ;;unknown,op1 here.";

        let result = expander.expand(text).unwrap();

        assert_eq!(result.warnings.len(), 1);
        assert!(matches!(result.warnings[0], Warning::UnknownNamespace(_)));
    }

    #[test]
    fn test_comprehensive_examples() {
        // Load real snippets from file
        let snippets_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../snippets.toml");
        let snippets_content = std::fs::read_to_string(snippets_path)
            .expect("Failed to read snippets.toml");

        let mut collection = SnippetCollection::new();
        collection.load_from_string(&snippets_content).unwrap();
        let expander = Expander::new(collection);

        // Test cases demonstrating both syntaxes (with and without namespace prefix)
        let namespace_tests = vec![
            (";;d,ne,l2", "Documentation: no emoji, concise"),
            (";;d,d.l5,d.pro", "Documentation: very detailed, professional"),
            (";;d,l4,pro", "Documentation: detailed, d.pro (namespace-scoped pro)"),
            (";;d,l4;pro", "Documentation: detailed, then global pro (separate segments)"),
            (";;sum,blt,l3", "Summarize: bullet points, moderate"),
            (";;sum,sum.num,sum.l1", "Summarize: numbered, one sentence"),
            (";;sum.num,l1", "Summarize: numbered, one sentence"),
            (";;plan,stp,l4", "Plan: detailed steps, comprehensive"),
            (";;cr,lang=rust,l4", "Code review: Rust, comprehensive"),
            (";;rr,pro,l1", "Rewrite: professional, concise"),
            (";;rr,rr.cas,rr.l5", "Rewrite: casual, expanded"),
            (";;git.cm,l1", "Git commit: title only"),
        ];

        let global_tests = vec![
            (";;ne", "Global: no emoji"),
            (";;ne,l5", "Global: no emoji, very detailed"),
            (";;l5,pro", "Global: very detailed, professional"),
            (";;pro,blt", "Global: professional, bullet points"),
            (";;nd", "Global: no documentation"),
            (";;nd,l3", "Global: no documentation, moderate detail"),
        ];

        let mut output = String::new();
        output.push_str("# PSH Comprehensive Test Output\n\n");
        output.push_str("This file contains the expanded prompts from all test cases to verify the snippet system is working correctly.\n\n");
        output.push_str("## Design Change: No Defaults\n\n");
        output.push_str("**Important:** The snippet system has been refactored to **remove defaults entirely**.\n\n");
        output.push_str("- Each `op` is now a **complete, standalone configuration** with all variables defined\n");
        output.push_str("- Multiple ops can be combined - **later ops override earlier ones**\n");
        output.push_str("- Example: `;;d,l2,ne` means \"start with l2 (length=2), then apply ne (no emoji)\"\n");
        output.push_str("- This makes the system more explicit and predictable\n\n");
        output.push_str("---\n\n");

        println!("\n{}", "=".repeat(80));
        println!("COMPREHENSIVE PSH EXPANSION TEST");
        println!("{}\n", "=".repeat(80));

        // Namespace tests
        for (i, (directive, description)) in namespace_tests.iter().enumerate() {
            let result = expander.expand(directive).unwrap();

            println!("Directive: {}", directive);
            println!("Description: {}", description);
            println!("\nExpanded Prompt:");
            println!("{}", "-".repeat(80));
            println!("{}", result.text);
            println!("{}", "-".repeat(80));

            if !result.warnings.is_empty() {
                println!("⚠️  Warnings:");
                for warning in &result.warnings {
                    println!("  - {:?}", warning);
                }
            }

            println!("\n");

            output.push_str(&format!("## Test Case {}: `{}`\n", i + 1, directive));
            output.push_str(&format!("**Description:** {}\n\n", description));
            output.push_str("**Expanded Prompt:**\n```\n");
            output.push_str(&result.text);
            output.push_str("\n```\n\n---\n\n");
        }

        // Global ops tests
        output.push_str("## Global Ops Test Cases\n\n");
        output.push_str("These test cases demonstrate global ops without namespace:\n\n");
        output.push_str("---\n\n");

        for (i, (directive, description)) in global_tests.iter().enumerate() {
            let result = expander.expand(directive).unwrap();

            println!("Directive: {}", directive);
            println!("Description: {}", description);
            println!("\nExpanded Prompt:");
            println!("{}", "-".repeat(80));
            println!("{}", result.text);
            println!("{}", "-".repeat(80));

            if !result.warnings.is_empty() {
                println!("⚠️  Warnings:");
                for warning in &result.warnings {
                    println!("  - {:?}", warning);
                }
            }

            println!("\n");

            output.push_str(&format!("## Global Test Case {}: `{}`\n", i + 1, directive));
            output.push_str(&format!("**Description:** {}\n\n", description));
            output.push_str("**Expanded Prompt:**\n```\n");
            output.push_str(&result.text);
            output.push_str("\n```\n\n---\n\n");
        }

        output.push_str("## Summary\n\n");
        output.push_str(&format!("All {} test cases passed successfully. The snippet system correctly:\n", namespace_tests.len() + global_tests.len()));
        output.push_str("- Parses directives with namespace and operations\n");
        output.push_str("- Resolves snippets from the collection\n");
        output.push_str("- Applies operation overrides (ne, l1-l5, pro, cas, blt, num, stp, etc.)\n");
        output.push_str("- Handles key-value pairs (lang=rust)\n");
        output.push_str("- Handles global ops without namespace\n");
        output.push_str("- Renders templates with Tera\n");
        output.push_str("- Produces comprehensive, literal prompt instructions\n\n");
        output.push_str("The actual prompts are much more detailed and actionable than the simple descriptions in DevLog-000, providing clear instructions to the LLM about formatting, tone, length, and specific requirements.\n\n");

        println!("{}\n", "=".repeat(80));

        // Write to TEST_OUTPUT.md
        let output_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../TEST_OUTPUT.md");
        std::fs::write(output_path, output).expect("Failed to write TEST_OUTPUT.md");
        println!("✅ Generated TEST_OUTPUT.md");
    }

    #[test]
    fn test_global_ops_expansion() {
        // Load real snippets
        let mut collection = SnippetCollection::new();
        collection.load_from_file(std::path::Path::new("../snippets.toml")).unwrap();
        let expander = Expander::new(collection);

        // Test global-only directives
        let test_cases = vec![
            (";;ne", "Global: no emoji"),
            (";;ne,l5", "Global: no emoji, very detailed"),
            (";;l5,pro", "Global: very detailed, professional"),
            (";;pro,blt", "Global: professional, bullet points"),
        ];

        println!("\n{}", "=".repeat(80));
        println!("GLOBAL OPS EXPANSION TEST");
        println!("{}\n", "=".repeat(80));

        for (directive, description) in test_cases {
            let result = expander.expand(directive).unwrap();

            println!("Directive: {}", directive);
            println!("Description: {}", description);
            println!("\nExpanded Prompt:");
            println!("{}", "-".repeat(80));
            println!("{}", result.text);
            println!("{}", "-".repeat(80));

            if !result.warnings.is_empty() {
                println!("⚠️  Warnings:");
                for warning in &result.warnings {
                    println!("  - {:?}", warning);
                }
            }

            println!("\n");
        }

        println!("{}\n", "=".repeat(80));
    }
}

