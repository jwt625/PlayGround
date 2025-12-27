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
fn parse_segment(segment: &str) -> Option<Segment> {
    let parts: Vec<&str> = segment.split(',').collect();

    if parts.is_empty() {
        return None;
    }

    let namespace = parts[0].trim().to_string();

    if namespace.is_empty() {
        return None;
    }

    let operations: Vec<Operation> = parts[1..]
        .iter()
        .filter_map(|&part| parse_operation(part.trim()))
        .collect();

    Some(Segment {
        namespace,
        operations,
    })
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
}

