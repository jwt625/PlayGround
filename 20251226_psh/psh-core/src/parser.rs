//! Directive parser for psh syntax
//!
//! Parses directives in the form: `;;namespace,op1,op2;namespace2,key=value`

use thiserror::Error;

/// Errors that can occur during parsing
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParseError {
    /// Invalid directive syntax
    #[error("Invalid directive syntax: {0}")]
    InvalidSyntax(String),
}

/// A parsed directive containing one or more segments
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Directive {
    /// The original directive text (including ;;)
    pub raw: String,
    /// The parsed segments
    pub segments: Vec<Segment>,
    /// Start position in the original text
    pub start: usize,
    /// End position in the original text
    pub end: usize,
}

/// A segment within a directive (namespace + operations)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// The namespace path (e.g., "d", "py.venv", "git.cm")
    pub namespace: String,
    /// Operations or key-value pairs to apply
    pub operations: Vec<Operation>,
}

/// An operation or key-value pair
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    /// A short operation code (e.g., "ne", "l5")
    Op(String),
    /// A key-value pair (e.g., "len=5")
    KeyValue {
        /// The key
        key: String,
        /// The value
        value: String,
    },
}

/// Parse all directives from input text
///
/// # Examples
///
/// ```
/// use psh_core::parse_directives;
///
/// let text = "Please ;;d,ne,l5 explain this code";
/// let directives = parse_directives(text);
/// assert_eq!(directives.len(), 1);
/// ```
#[must_use]
pub fn parse_directives(text: &str) -> Vec<Directive> {
    let mut directives = Vec::new();
    let mut pos = 0;

    while pos < text.len() {
        // Look for ;; sentinel
        if let Some(start) = find_sentinel(text, pos) {
            // Parse the directive
            if let Some((directive, end)) = parse_directive(text, start) {
                directives.push(directive);
                pos = end;
            } else {
                // Skip this ;; and continue
                pos = start + 2;
            }
        } else {
            break;
        }
    }

    directives
}

/// Find the next unescaped ;; sentinel
fn find_sentinel(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = start;

    while i + 1 < bytes.len() {
        if bytes[i] == b';' && bytes[i + 1] == b';' {
            // Check if it's escaped
            if i > 0 && bytes[i - 1] == b'\\' {
                // Escaped, skip
                i += 2;
                continue;
            }
            return Some(i);
        }
        i += 1;
    }

    None
}

/// Parse a single directive starting at the ;; sentinel
fn parse_directive(text: &str, start: usize) -> Option<(Directive, usize)> {
    // Skip the ;; sentinel
    let pos = start + 2;

    // Find the end of the directive (next whitespace or end of string)
    let end = find_directive_end(text, pos);
    
    if end == pos {
        // Empty directive
        return None;
    }

    let directive_text = &text[pos..end];
    
    // Parse segments separated by ;
    let segments: Vec<Segment> = directive_text
        .split(';')
        .filter_map(parse_segment)
        .collect();

    if segments.is_empty() {
        return None;
    }

    Some((
        Directive {
            raw: text[start..end].to_string(),
            segments,
            start,
            end,
        },
        end,
    ))
}

/// Find the end of a directive (next whitespace or end of string)
fn find_directive_end(text: &str, start: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = start;

    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            return i;
        }
        i += 1;
    }

    i
}

/// Parse a segment (namespace,op1,op2,key=value)
///
/// Supports two formats:
/// 1. With namespace: `d,ne,l5` -> namespace=`d`, ops=`["ne", "l5"]`
/// 2. Global ops only: `ne,l5` -> namespace=`_global`, ops=`["ne", "l5"]`
///
/// A segment is treated as global-only if the first element looks like an operation
/// (contains '=' for key-value, or is a short code without dots that could be a namespace)
fn parse_segment(segment: &str) -> Option<Segment> {
    let parts: Vec<&str> = segment.split(',').collect();

    if parts.is_empty() {
        return None;
    }

    let first = parts[0].trim();

    if first.is_empty() {
        return None;
    }

    // Determine if this is a global-only segment (no namespace)
    // It's global-only if the first element looks like an operation:
    // - Contains '=' (key-value pair)
    // - Is a short code that looks like a known op pattern (l1-l5, ne, e, pro, cas, tec, blt, num, etc.)
    let is_global_only = first.contains('=') || is_likely_op(first);

    let (namespace, op_start_idx) = if is_global_only {
        // No namespace, all parts are operations
        ("_global".to_string(), 0)
    } else {
        // First part is namespace, rest are operations
        (first.to_string(), 1)
    };

    let operations: Vec<Operation> = parts[op_start_idx..]
        .iter()
        .filter_map(|&part| parse_operation(part.trim()))
        .collect();

    Some(Segment {
        namespace,
        operations,
    })
}

/// Check if a string looks like an operation rather than a namespace
///
/// Operations are typically:
/// - Short codes: l1-l5, ne, e, pro, cas, tec, blt, num, stp, ask, base
/// - Namespace-scoped ops: d.ne, sum.l1, etc. (but these have dots)
///
/// Namespaces are typically:
/// - Single letters or short words: d, sum, plan, cr, rr, qa, fmt
/// - Dot-separated paths: py.venv, git.cm
///
/// Heuristic: If it's a very short code (1-4 chars) that matches common op patterns,
/// or if it contains a dot (namespace-scoped op), treat it as an op.
/// Otherwise, treat it as a namespace.
fn is_likely_op(s: &str) -> bool {
    // Known global op patterns
    const KNOWN_OPS: &[&str] = &[
        "l1", "l2", "l3", "l4", "l5",  // Length levels
        "ne", "e",                      // Emoji control
        "pro", "cas", "tec",            // Tone modifiers
        "blt", "num",                   // Format modifiers
        "stp", "ask",                   // Other modifiers
        "nd",                           // Documentation control
        "base",                         // Base op
    ];

    // Check if it's a known op
    if KNOWN_OPS.contains(&s) {
        return true;
    }

    // If it contains a dot and is not a simple namespace (like "py.venv"),
    // it's likely a namespace-scoped op (like "d.ne")
    // We distinguish by checking if the part after the dot looks like an op
    if let Some(dot_pos) = s.find('.') {
        let after_dot = &s[dot_pos + 1..];
        return KNOWN_OPS.contains(&after_dot);
    }

    false
}

/// Parse an operation or key-value pair
fn parse_operation(op: &str) -> Option<Operation> {
    if op.is_empty() {
        return None;
    }

    // Check if it's a key=value pair
    if let Some(eq_pos) = op.find('=') {
        let key = op[..eq_pos].trim().to_string();
        let value = op[eq_pos + 1..].trim().to_string();

        if key.is_empty() {
            return None;
        }

        Some(Operation::KeyValue { key, value })
    } else {
        // It's an op code
        Some(Operation::Op(op.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_directive() {
        let text = ";;d,ne,l5";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments.len(), 1);
        assert_eq!(directives[0].segments[0].namespace, "d");
        assert_eq!(directives[0].segments[0].operations.len(), 2);
    }

    #[test]
    fn test_parse_multiple_segments() {
        let text = ";;d,ne,l5;py.venv";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments.len(), 2);
        assert_eq!(directives[0].segments[0].namespace, "d");
        assert_eq!(directives[0].segments[1].namespace, "py.venv");
    }

    #[test]
    fn test_parse_key_value() {
        let text = ";;cr,lang=py,l4";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments[0].operations.len(), 2);

        match &directives[0].segments[0].operations[0] {
            Operation::KeyValue { key, value } => {
                assert_eq!(key, "lang");
                assert_eq!(value, "py");
            }
            _ => panic!("Expected KeyValue"),
        }
    }

    #[test]
    fn test_parse_multiple_directives() {
        let text = "Please ;;d,ne,l5 explain ;;sum,blt this code";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 2);
        assert_eq!(directives[0].segments[0].namespace, "d");
        assert_eq!(directives[1].segments[0].namespace, "sum");
    }

    #[test]
    fn test_escaped_sentinel() {
        let text = "Use \\;; for literal semicolons ;;d,ne";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments[0].namespace, "d");
    }

    #[test]
    fn test_empty_directive() {
        let text = ";; ;;d,ne";
        let directives = parse_directives(text);

        // Empty directive should be skipped
        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments[0].namespace, "d");
    }

    #[test]
    fn test_directive_positions() {
        let text = "Start ;;d,ne end";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].start, 6);
        assert_eq!(directives[0].end, 12);
    }

    #[test]
    fn test_global_only_single_op() {
        let text = ";;ne";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments.len(), 1);
        assert_eq!(directives[0].segments[0].namespace, "_global");
        assert_eq!(directives[0].segments[0].operations.len(), 1);

        match &directives[0].segments[0].operations[0] {
            Operation::Op(op) => assert_eq!(op, "ne"),
            _ => panic!("Expected Op"),
        }
    }

    #[test]
    fn test_global_only_multiple_ops() {
        let text = ";;ne,l5";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments.len(), 1);
        assert_eq!(directives[0].segments[0].namespace, "_global");
        assert_eq!(directives[0].segments[0].operations.len(), 2);

        match &directives[0].segments[0].operations[0] {
            Operation::Op(op) => assert_eq!(op, "ne"),
            _ => panic!("Expected Op"),
        }

        match &directives[0].segments[0].operations[1] {
            Operation::Op(op) => assert_eq!(op, "l5"),
            _ => panic!("Expected Op"),
        }
    }

    #[test]
    fn test_global_only_with_key_value() {
        let text = ";;l5,pro,custom_key=value";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments.len(), 1);
        assert_eq!(directives[0].segments[0].namespace, "_global");
        assert_eq!(directives[0].segments[0].operations.len(), 3);
    }

    #[test]
    fn test_namespace_vs_global_distinction() {
        // "d,ne" should be namespace "d" with op "ne"
        let text1 = ";;d,ne";
        let directives1 = parse_directives(text1);
        assert_eq!(directives1[0].segments[0].namespace, "d");
        assert_eq!(directives1[0].segments[0].operations.len(), 1);

        // "ne" should be global with op "ne"
        let text2 = ";;ne";
        let directives2 = parse_directives(text2);
        assert_eq!(directives2[0].segments[0].namespace, "_global");
        assert_eq!(directives2[0].segments[0].operations.len(), 1);
    }

    #[test]
    fn test_namespace_scoped_op_in_global() {
        // "d.ne" should be treated as global with namespace-scoped op "d.ne"
        let text = ";;d.ne,l5";
        let directives = parse_directives(text);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].segments.len(), 1);
        assert_eq!(directives[0].segments[0].namespace, "_global");
        assert_eq!(directives[0].segments[0].operations.len(), 2);

        match &directives[0].segments[0].operations[0] {
            Operation::Op(op) => assert_eq!(op, "d.ne"),
            _ => panic!("Expected Op"),
        }
    }
}

