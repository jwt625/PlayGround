# psh — Project Scoping, Design, and Planning Document

## 1. Overview

**psh** is a macOS-first, OS-level prompt pre-parser for LLM interactions.  
It allows users to embed highly concise, information-dense directives directly into natural language text. These directives are expanded into full prompt snippets before being sent to an LLM.

The system is designed to behave like a shell or compiler for prompts:
- Short-hand, composable syntax
- Deterministic parsing and expansion
- OS-level integration
- Local-first, privacy-preserving operation

The MVP focuses exclusively on macOS, with architecture decisions that allow future cross-platform expansion.

---

## 2. Goals and Non-Goals

### Goals (MVP)
- OS-level invocation via global hotkey
- Overlay UI with preview → apply workflow
- In-place replacement of psh directives with expanded prompt snippets
- Extremely concise directive syntax
- Local-only configuration and usage tracking
- User-editable prompt snippet system

### Non-Goals (MVP)
- Cross-device sync
- Cloud-based telemetry
- Semantic conflict resolution between snippets
- Cross-platform UI parity
- Prompt execution or LLM API integration

---

## 3. High-Level Architecture

### 3.1 Core + Adapter Model

The system is split into two layers:

#### Core Engine (Cross-Platform)
- Written in Rust
- Responsibilities:
  - Directive parsing
  - Snippet resolution and composition
  - Template rendering
  - Usage statistics aggregation
  - Configuration parsing and validation

The core is exposed via a stable C-compatible interface for Swift integration.

#### macOS Adapter (MVP)
- Written in Swift / SwiftUI
- Responsibilities:
  - Global hotkey handling
  - Accessibility API integration (read/write focused text)
  - Overlay UI (preview, warnings, apply)
  - Preferences UI
  - Local storage coordination

---

## 4. Invocation Model

### 4.1 Activation
- Global keyboard shortcut (configurable)
- Reads current focused text input
- Presents overlay UI with:
  - Detected directives
  - Expanded preview
  - Warnings (unknown ops, namespaces)
  - Apply / Cancel controls
- User may opt out of confirmation (“Do not ask again”)

---

## 5. Directive Language Specification (MVP)

### 5.1 Sentinel and Escaping
- All directives begin with the sentinel: `;;`
- A directive token ends at the next whitespace
- Literal `;;` may be written as `\;;`

### 5.2 Multiple Directives
- Multiple `;;` directives are allowed anywhere in the text
- Each directive is parsed independently
- Directives are replaced in-place with their expansions

---

## 6. Directive Grammar

### 6.1 Token Structure

```
;; <segment> ( ; <segment> )*
```

Each segment targets a namespace or namespace path.

### 6.2 Segment Structure

```
<namespace_path> , <op_or_kv> ( , <op_or_kv> )*
```

Where:
- `namespace_path` uses dot notation (e.g., `d`, `py.venv`)
- `op_or_kv` is either:
  - a short operation code (e.g., `ne`, `l5`)
  - a key-value pair using `key=value`

### 6.3 Examples

```
;;d,ne,len=5
;;d,ne,l5;py.venv
;;s,ne;d,l3
```

---

## 7. Semantics and Resolution Rules

### 7.1 Expansion Model
- psh does not interpret user intent beyond snippet expansion
- Each namespace resolves to a base snippet
- Ops and key-value pairs apply overrides to that snippet
- The final expanded text replaces the directive token

### 7.2 Conflict Handling
- No semantic conflict resolution in MVP
- If multiple ops override the same property:
  - Last one wins
- User is responsible for avoiding contradictory directives

### 7.3 Unknown Elements
- Unknown namespace, op, or key:
  - Ignored
  - Warning displayed in overlay UI
- Expansion proceeds for known elements

---

## 8. Snippet System Design

### 8.1 Conceptual Model
Prompt snippets behave like composable functions or objects:
- Base snippet selected by namespace
- Ops and key-value pairs modify parameters
- Final rendering produces text

### 8.2 Required Snippet Metadata
Each snippet must define:
- `id`
- `namespace`
- `template`
- Supported ops
- Supported keys and default values
- Tags (task, language, tone, etc.)

### 8.3 Storage Format
- User-editable file (TOML recommended)
- Compiled into an internal representation at runtime
- Hot-reload supported where possible

---

## 9. Overlay UI Requirements (MVP)

- Highlight detected directives
- Show expanded preview text
- Show warnings for unknown elements
- Allow apply / cancel
- Provide shortcut to snippet search/help
- Support “Do not ask again” toggle (global or per-app)

---

## 10. Autocomplete and Help

Given the dense syntax, discoverability is mandatory:

- Keyboard-invoked search palette
- Fuzzy search by:
  - Namespace
  - Op code
  - Description
  - Tags
- Inline suggestions for unknown ops
- Documentation panel per snippet

---

## 11. Usage Tracking (Local-Only)

### 11.1 Default Behavior
- Track directive usage counts
- Track last-used timestamps
- No full-text prompt storage by default

### 11.2 Opt-In Advanced Tracking
- User may opt in to storing:
  - Full original prompt
  - Expanded prompt
  - Directive tokens
- Data is stored locally only
- Clear-all and per-app exclusion controls required

### 11.3 Future Use
- Detect frequently repeated patterns
- Recommend creation of new snippets
- Suggest consolidation of common ops

---

## 12. External Prompt Sources

### MVP Position
- No live community sharing platform
- Optional import of curated external prompt collections
- Imported prompts must be normalized into psh snippet schema
- Attribution and license compliance required

---

## 13. Security and Privacy

- Local-first by design
- No network access required for core functionality
- Explicit opt-in for any persistent prompt logging
- Clear and auditable configuration files

---

## 14. Open Future Directions (Post-MVP)

- Cross-platform adapters (Linux, Windows)
- Per-app snippet defaults
- Scoped directives
- Snippet inheritance and composition graphs
- Community snippet packs and sharing
- Semantic clustering and recommendation engine

---

## 15. Summary of Key Design Decisions

- macOS-only MVP
- Swift UI + Rust core
- Overlay-first UX
- `;;` as directive sentinel
- `;` for namespace chaining, `,` for ops/kv
- In-place replacement model
- Dense, information-rich syntax inspired by CLI tools
- Local-only data handling

This document defines the authoritative MVP scope and design contract for psh.


## Examples

1. ;;d,ne,l2
   Documentation-style response, no emoji, concise (length level 2).

2. ;;d,ne,l5
   Documentation-style response, no emoji, very detailed (length level 5).

3. ;;sum,blt,l3
   Summarize the input into bullet points with medium verbosity.

4. ;;plan,stp,l4
   Generate a step-by-step plan with relatively high detail.

5. ;;cr,lang=py,l4;d,ne
   Perform a Python code review with detailed feedback and no emoji in the explanation.

6. ;;rr,pro,l2
   Rewrite the text in a professional tone, concise length.

7. ;;git.cm,l1
   Generate a one-line git commit message.

8. ;;qa,ask,l5
   Ask high-leverage clarification questions before answering, in depth.

9. ;;py.venv
   Provide guidance related to Python virtual environments using default verbosity.

10. ;;fmt,md;d,l3
    Format the output as Markdown and use medium-length documentation style.
