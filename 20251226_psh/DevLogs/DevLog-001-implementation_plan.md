# DevLog-001: Implementation Plan and Task Breakdown

## Document Purpose

This document provides a detailed implementation plan for the psh MVP, breaking down the architecture into concrete tasks, dependencies, and deliverables.

## Implementation Strategy

### Phase 1: Core Engine (Rust)
Build the cross-platform parsing and expansion engine first, with CLI testing harness.

### Phase 2: macOS Adapter (Swift)
Implement OS integration, UI, and bridge to Rust core.

### Phase 3: Integration and Polish
End-to-end testing, starter snippets, documentation.

---

## Phase 1: Core Engine (Rust)

### 1.1 Project Setup
- [ ] Initialize Rust workspace with cargo
- [ ] Set up project structure:
  - `psh-core/` - Core parsing and expansion logic
  - `psh-ffi/` - C-compatible FFI layer for Swift
  - `psh-cli/` - CLI testing harness
- [ ] Configure build system for static library output
- [ ] Add dependencies:
  - `tera` - Template engine
  - `serde` - Serialization
  - `toml` - Configuration parsing
  - `notify` - File watching for hot-reload
  - `thiserror` - Error handling

### 1.2 Directive Parser
- [ ] Define AST structures:
  - `Directive` - Top-level parsed directive
  - `Segment` - Namespace + ops/kv pairs
  - `Operation` - Op code or key-value pair
- [ ] Implement tokenizer:
  - Detect `;;` sentinel
  - Handle escape sequences `\;;`
  - Split on `;` for segments
  - Split on `,` for namespace and ops
- [ ] Implement parser:
  - Parse namespace paths (dot notation)
  - Parse op codes
  - Parse key=value pairs
  - Error recovery for malformed directives
- [ ] Unit tests for parser edge cases

### 1.3 Snippet System
- [ ] Define snippet schema (TOML format):
  ```toml
  [[snippet]]
  id = "doc-style"
  namespace = "d"
  template = "..."
  
  [snippet.ops]
  ne = { emoji = "false" }
  l1 = { length_instruction = "Be extremely concise..." }
  
  [snippet.defaults]
  emoji = "true"
  length_instruction = "Use moderate detail..."
  ```
- [ ] Implement snippet loader:
  - Parse TOML files from config directory
  - Validate schema
  - Build namespace index
  - Build op lookup tables
- [ ] Implement snippet resolver:
  - Match namespace to snippet
  - Apply ops to override variables
  - Apply key=value pairs
  - Collect unknown namespaces/ops for warnings
- [ ] Unit tests for resolution logic

### 1.4 Template Rendering
- [ ] Integrate Tera template engine
- [ ] Implement variable resolution:
  - Start with snippet defaults
  - Override with op-defined values
  - Override with key=value pairs (last wins)
- [ ] Implement template rendering with error handling
- [ ] Unit tests for rendering edge cases

### 1.5 Expansion Engine
- [ ] Implement full text expansion:
  - Find all `;;` directives in input text
  - Parse each directive
  - Resolve and render each directive
  - Replace directives in-place with expansions
  - Preserve original text structure
- [ ] Collect warnings (unknown elements)
- [ ] Integration tests with realistic examples

### 1.6 Configuration Management
- [ ] Define configuration schema:
  - Snippet directories
  - Hot-reload settings
  - Usage tracking preferences
- [ ] Implement config loader with defaults
- [ ] Implement file watcher for snippet hot-reload
- [ ] Unit tests for config parsing

### 1.7 Usage Tracking
- [ ] Define usage data structures:
  - Directive usage counts
  - Last-used timestamps
  - Optional: full prompt storage (opt-in)
- [ ] Implement local storage (SQLite or JSON)
- [ ] Implement aggregation queries
- [ ] Privacy controls (clear-all, per-app exclusion)

### 1.8 FFI Layer
- [ ] Define C-compatible interface:
  - `psh_init()` - Initialize engine
  - `psh_expand()` - Expand text with directives
  - `psh_get_warnings()` - Retrieve warnings
  - `psh_reload_snippets()` - Manual reload
  - `psh_shutdown()` - Cleanup
- [ ] Implement FFI wrappers with proper memory management
- [ ] Generate C header file for Swift import
- [ ] Test FFI from C test harness

### 1.9 CLI Testing Harness
- [ ] Build CLI tool for testing:
  - Read input from stdin or file
  - Expand directives
  - Output expanded text
  - Display warnings
- [ ] Use for manual testing and debugging

---

## Phase 2: macOS Adapter (Swift)

### 2.1 Project Setup
- [ ] Create Xcode project for macOS app
- [ ] Configure Swift package dependencies
- [ ] Import Rust static library and headers
- [ ] Set up SwiftUI app structure
- [ ] Configure app entitlements:
  - Accessibility API access
  - Global hotkey registration

### 2.2 Rust Bridge
- [ ] Create Swift wrapper for FFI:
  - Initialize psh-core on app launch
  - Call expansion functions
  - Handle memory management (strings, errors)
  - Convert warnings to Swift types
- [ ] Unit tests for bridge layer

### 2.3 Accessibility Integration
- [ ] Request accessibility permissions
- [ ] Implement focused text reading:
  - Get active application
  - Get focused UI element
  - Read text content via AX API
- [ ] Implement focused text writing:
  - Replace text in focused element
  - Preserve cursor position where possible
- [ ] Handle edge cases (unsupported apps, permissions denied)

### 2.4 Global Hotkey
- [ ] Implement global hotkey registration (default: Cmd+Shift+;)
- [ ] Handle hotkey events
- [ ] Trigger expansion workflow on activation
- [ ] Allow hotkey customization in preferences

### 2.5 Overlay UI
- [ ] Design overlay window:
  - Floating, always-on-top window
  - Position near cursor or focused element
  - Highlight detected directives
  - Show expanded preview (diff view)
  - Display warnings prominently
  - Apply / Cancel buttons
  - "Do not ask again" toggle
- [ ] Implement SwiftUI views
- [ ] Handle keyboard navigation (Enter = Apply, Esc = Cancel)
- [ ] Animate transitions

### 2.6 Preferences UI
- [ ] Create preferences window:
  - Hotkey customization
  - Snippet directory location
  - Usage tracking opt-in/opt-out
  - Clear usage data button
  - Manual snippet reload button
- [ ] Implement settings persistence (UserDefaults)

### 2.7 Snippet Search/Help UI
- [ ] Implement search palette:
  - Fuzzy search by namespace, op, description, tags
  - Display snippet documentation
  - Show available ops and keys
  - Insert directive template on selection
- [ ] Keyboard shortcut to invoke (Cmd+Shift+/)

---

## Phase 3: Integration and Polish

### 3.1 Starter Snippet Set
- [ ] Create curated snippets based on examples:
  - `d` - Documentation style
  - `sum` - Summarization
  - `plan` - Planning
  - `cr` - Code review
  - `rr` - Rewrite
  - `git.cm` - Git commit messages
  - `qa` - Question asking
  - `py.venv` - Python venv guidance
  - `fmt` - Formatting
- [ ] Define ops:
  - `ne` - No emoji
  - `l1` through `l5` - Length levels
  - `blt` - Bullet points
  - `stp` - Step-by-step
  - `pro` - Professional tone
  - `ask` - Ask questions first
- [ ] Write templates with Tera syntax
- [ ] Document each snippet with descriptions and examples

### 3.2 End-to-End Testing
- [ ] Test full workflow:
  - Hotkey activation
  - Text reading from various apps
  - Directive parsing and expansion
  - Preview display
  - Text replacement
- [ ] Test edge cases:
  - Multiple directives in one text
  - Escaped `\;;`
  - Unknown namespaces/ops
  - Empty directives
  - Very long expansions
- [ ] Test hot-reload functionality
- [ ] Test usage tracking

### 3.3 Error Handling and UX Polish
- [ ] Graceful degradation for accessibility failures
- [ ] Clear error messages in overlay
- [ ] Loading states for slow operations
- [ ] Accessibility support (VoiceOver compatibility)
- [ ] App icon and branding

### 3.4 Documentation
- [ ] User guide:
  - Installation instructions
  - First-time setup
  - Basic usage examples
  - Snippet customization guide
- [ ] Developer documentation:
  - Architecture overview
  - Adding new snippets
  - Building from source
- [ ] README with quick start

---

## Dependencies and Critical Path

### Critical Path
1. Directive Parser → Snippet System → Expansion Engine (Core)
2. FFI Layer (Bridge)
3. Accessibility Integration + Global Hotkey (macOS)
4. Overlay UI (macOS)
5. Integration Testing

### Parallel Work Opportunities
- CLI harness can be built alongside FFI
- Preferences UI can be built while core is in development
- Starter snippets can be written early and refined later
- Documentation can be drafted in parallel

---

## Risk Mitigation

### Technical Risks
1. **Accessibility API limitations**: Some apps may not expose text properly
   - Mitigation: Test early with target apps, document limitations
2. **FFI complexity**: Memory management across Rust/Swift boundary
   - Mitigation: Use established patterns, thorough testing
3. **Hot-reload stability**: File watching can be flaky
   - Mitigation: Provide manual reload, test extensively

### Scope Risks
1. **Feature creep**: Temptation to add non-MVP features
   - Mitigation: Strict adherence to DevLog-000 scope
2. **Snippet design complexity**: Balancing flexibility and simplicity
   - Mitigation: Start with simple examples, iterate based on usage

---

## Success Criteria

MVP is complete when:
- [ ] User can invoke psh via global hotkey
- [ ] Directives are correctly parsed and expanded
- [ ] Overlay shows preview and warnings
- [ ] Text is replaced in-place in focused app
- [ ] Starter snippet set is functional
- [ ] Hot-reload works for snippet changes
- [ ] Basic usage tracking is operational
- [ ] User documentation is complete

---

## Next Steps

1. Begin Phase 1.1: Set up Rust workspace
2. Implement directive parser (Phase 1.2)
3. Build snippet system (Phase 1.3)
4. Continue through phases sequentially

This plan will be updated as implementation progresses and new insights emerge.

