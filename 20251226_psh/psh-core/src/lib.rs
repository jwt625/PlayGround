//! psh-core: Core parsing and expansion engine for psh
//!
//! This crate provides the cross-platform directive parsing, snippet resolution,
//! and template rendering functionality for the psh prompt pre-parser.

pub mod expander;
pub mod parser;
pub mod resolver;
pub mod snippet;

pub use expander::{ExpandError, ExpandResult, Expander};
pub use parser::{Directive, Operation, ParseError, Segment, parse_directives};
pub use resolver::{ResolvedDirective, Resolver, ResolverError, Warning};
pub use snippet::{Snippet, SnippetCollection, SnippetError};
