# DevLog-001: Implementation Plan and Task Breakdown

## Document Purpose

This document provides a detailed implementation plan for the psh MVP, breaking down the architecture into concrete tasks, dependencies, and deliverables.

## Development Standards

### Rust Tooling
- **Package Manager**: Cargo (built-in)
- **Linter**: Clippy (official, ~600 lints)
- **Formatter**: rustfmt (official, opinionated)
- **Type Checking**: Built into Rust compiler (mandatory)
- **Additional Tools**:
  - `cargo-watch` - Auto-rebuild on file changes
  - `cargo-audit` - Security vulnerability scanning

### Development Workflow
```bash
cargo check          # Fast type checking
cargo clippy         # Linting
cargo fmt            # Format code
cargo test           # Run tests
cargo build          # Build binary
```

### Pre-Commit Standards
- `cargo fmt --check` - Verify formatting
- `cargo clippy -- -D warnings` - No warnings allowed
- `cargo test` - All tests pass
- `cargo audit` - No known vulnerabilities

### Code Practices
- Use Rust 2021 edition
- Enable `clippy::pedantic` and `clippy::nursery` lints
- Maximum line width: 100 characters
- Comprehensive error handling with `thiserror`
- Document public APIs with doc comments (`///`)
- Unit tests co-located with code (`#[cfg(test)]`)
- Integration tests in `tests/` directory

### Project Management
- Track progress via DevLog updates
- One feature per commit with clear messages
- Tag milestones: `v0.1.0-phase1`, `v0.2.0-phase2`, etc.
- Document breaking changes in commit messages

---

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
- [x] Initialize Rust workspace with cargo
- [x] Set up project structure:
  - `psh-core/` - Core parsing and expansion logic
  - `psh-ffi/` - C-compatible FFI layer for Swift (deferred)
  - `psh-cli/` - CLI testing harness (removed - not needed)
- [x] Configure build system for library output
- [x] Add dependencies:
  - `tera` - Template engine
  - `serde` - Serialization
  - `toml` - Configuration parsing
  - `thiserror` - Error handling

### 1.2 Directive Parser
- [x] Define AST structures:
  - `Directive` - Top-level parsed directive
  - `Segment` - Namespace + ops/kv pairs
  - `Operation` - Op code or key-value pair
- [x] Implement tokenizer:
  - Detect `;;` sentinel
  - Handle escape sequences `;;;;` → `;;`
  - Split on `;` for segments
  - Split on `,` for namespace and ops
- [x] Implement parser:
  - Parse namespace paths (dot notation)
  - Parse op codes
  - Parse key=value pairs
  - Track positions for in-place replacement
- [x] Unit tests for parser edge cases (7 tests)

### 1.3 Snippet System
- [x] Define snippet schema (TOML format) - **Design change: removed defaults**
  ```toml
  [[snippet]]
  id = "doc-style"
  namespace = "d"
  template = "..."

  [snippet.ops.l2]
  length_instruction = "Be concise..."
  emoji_instruction = "Use emoji sparingly..."
  tone_instruction = "Professional tone..."

  [snippet.ops.ne]
  emoji_instruction = "No emoji..."
  ```
  Each op is a complete configuration. Multiple ops combine (later overrides earlier).
- [x] Implement snippet loader:
  - Parse TOML files with serde
  - Validate schema
  - Build namespace index
- [x] Implement snippet resolver:
  - Match namespace to snippet
  - Apply ops to build context (later ops override earlier)
  - Apply key=value pairs (last wins)
  - Collect unknown namespaces/ops/keys for warnings
- [x] Unit tests for resolution logic (3 tests)

### 1.4 Template Rendering
- [x] Integrate Tera template engine
- [x] Implement variable resolution:
  - Build context from ops (no defaults)
  - Apply ops in order (later overrides earlier)
  - Apply key=value pairs (last wins)
- [x] Implement template rendering with Tera.one_off
- [x] Unit tests for rendering edge cases (6 tests)

### 1.5 Expansion Engine
- [x] Implement full text expansion:
  - Find all `;;` directives in input text
  - Parse each directive
  - Resolve and render each directive
  - Replace directives in-place with expansions
  - Preserve original text structure
- [x] Collect warnings (unknown elements)
- [x] Integration tests with realistic examples (4 tests + 1 comprehensive)

### 1.6 Configuration Management (Deferred)
- [ ] Define configuration schema:
  - Snippet directories
  - Hot-reload settings
  - Usage tracking preferences
- [ ] Implement config loader with defaults
- [ ] Implement file watcher for snippet hot-reload
- [ ] Unit tests for config parsing

### 1.7 Usage Tracking (Deferred)
- [ ] Define usage data structures:
  - Directive usage counts
  - Last-used timestamps
  - Optional: full prompt storage (opt-in)
- [ ] Implement local storage (SQLite or JSON)
- [ ] Implement aggregation queries
- [ ] Privacy controls (clear-all, per-app exclusion)

### 1.8 FFI Layer
- [x] Define C-compatible interface:
  - `psh_init()` - Initialize engine
  - `psh_expand()` - Expand text with directives
  - `psh_reload_snippets()` - Manual reload
  - `psh_shutdown()` - Cleanup
  - `psh_free_result()` - Free expansion results
- [x] Implement FFI wrappers with proper memory management
- [x] Generate C header file for Swift import (`psh.h`)
- [x] Test FFI layer (2 tests passing)

---

## Phase 1 Status: COMPLETE

**Completion Date**: 2025-12-27

**Summary**: Core Rust engine fully implemented and tested. All 26 tests pass with zero clippy warnings.

**Key Design Decisions**:
1. **Initial Design (2025-12-27)**: Removed snippet defaults entirely. Each op was a complete, standalone configuration. Multiple ops combined with later ops overriding earlier ones.

2. **Revised Design (2025-12-27)**: Implemented hierarchical op system with base defaults and single-purpose ops:
   - Added `base` ops at global and namespace levels to provide defaults
   - Made ops single-purpose: length ops only set length, emoji ops only set emoji, tone ops only set tone
   - Resolver applies ops in order: global base → namespace base → user ops → key-value pairs
   - Supports namespace-scoped op syntax (e.g., `d.ne` and `ne` both work)

**Critical Bugfix (2025-12-27)**: Fixed ops incorrectly setting multiple unrelated variables. Length ops (l1-l5) were setting emoji and tone instructions, causing incorrect behavior. For example, `;;d,l5,pro` would produce "use emoji sparingly" instead of "no emoji" (the namespace default). Solution: each op now has single responsibility, with base ops providing complete defaults.

**Deliverables**:
- `psh-core/src/parser.rs` - Directive parser with position tracking
- `psh-core/src/snippet.rs` - TOML-based snippet loader with global ops support
- `psh-core/src/resolver.rs` - Template resolver with base op application and namespace-scoped op support
- `psh-core/src/expander.rs` - Full text expansion engine
- `snippets.toml` - Comprehensive snippet library with hierarchical op system (d, sum, plan, cr, rr, git.cm)

**Test Coverage**:
- Parser: 7 tests
- Snippet: 3 tests
- Resolver: 9 tests (added namespace-scoped op tests)
- Expander: 4 tests
- Integration: 1 comprehensive test
- Doc tests: 1
- Total: 26 tests, all passing

**Next Steps**: Proceed to Phase 2 (macOS adapter) or refine snippet library.

---

## Phase 2: macOS Adapter (Swift)

### 2.1 Project Setup
- [x] Create Swift Package project for macOS app
- [x] Configure Swift package dependencies
- [x] Import Rust static library and headers
- [x] Set up SwiftUI app structure
- [x] Configure app as menu bar utility

### 2.2 Rust Bridge
- [x] Create Swift wrapper for FFI:
  - Initialize psh-core on app launch
  - Call expansion functions
  - Handle memory management (strings, errors)
  - Convert warnings to Swift types
- [x] System library module for C header bridging

### 2.3 Accessibility Integration
- [x] Request accessibility permissions
- [x] Implement focused text reading:
  - Get active application
  - Get focused UI element
  - Read text content via AX API
- [x] Implement focused text writing:
  - Replace text in focused element
- [x] Handle edge cases (unsupported apps, permissions denied)

### 2.4 Global Hotkey
- [x] Implement global hotkey registration (Cmd+Shift+;)
- [x] Handle hotkey events via Carbon Event Manager
- [x] Trigger expansion workflow on activation

### 2.5 Overlay UI
- [x] Design overlay window:
  - Floating, always-on-top window
  - Show expanded preview (diff view)
  - Display warnings prominently
  - Apply / Cancel buttons
  - Keyboard shortcuts (Enter = Apply, Esc = Cancel)
- [x] Implement SwiftUI views
- [x] Handle keyboard navigation

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

## Current Status

**Phase 1: COMPLETE** (2025-12-27)
- Core Rust engine fully functional
- 35 tests passing (26 core + 8 FFI + 1 doc), zero clippy warnings
- Comprehensive snippet library with hierarchical op system
- Design refined: base ops provide defaults, single-purpose ops override specific variables
- Namespace-scoped op syntax supported (e.g., `;;d,ne` and `;;d,d.ne` both work)
- **FFI Layer COMPLETE** (2025-12-27):
  - C-compatible bindings for Swift integration
  - Memory-safe wrapper functions with proper cleanup
  - C header file (`psh.h`) for bridging
  - Global mutex-protected expander instance
  - Comprehensive test coverage (8 tests covering all edge cases)

**Phase 2: macOS App COMPLETE** (2025-12-27)
- Swift Package Manager project structure
- Menu bar app (no dock icon, runs in background)
- **Core Components**:
  - `PshWrapper.swift` - Memory-safe Swift wrapper around C FFI
  - `AccessibilityManager.swift` - Read/write text via pasteboard (Cmd+A, Cmd+C, Cmd+V)
  - `HotkeyManager.swift` - Global hotkey (Cmd+Shift+;) via Carbon
  - `AppCoordinator.swift` - Main app coordinator with auto-apply support
  - `main.swift` - App entry point with menu bar integration
- **UI Components**:
  - `OverlayWindow.swift` - Preview overlay with config options
  - `ConfigWindow.swift` - Config/snippet browser window
  - `SnippetBrowserView.swift` - Snippet browser with search
  - `SnippetInfo.swift` - Snippet metadata parser
  - `UserPreferences.swift` - Settings persistence
- **Build system**: Automated Rust library compilation
- **Snippets**: Supports `~/.config/psh/snippets.toml` + bundled defaults
- **Permissions**: Auto-requests accessibility permissions
- **App builds successfully**: Debug and release builds working

**Implementation Details & Bug Fixes** (2025-12-27):
1. **Text Reading/Writing**: Simplified approach using system pasteboard
   - Send Cmd+A to select all text
   - Send Cmd+C to copy to pasteboard
   - Read text from pasteboard
   - For writing: put text on pasteboard, Cmd+A, Cmd+V
   - More reliable than AX API focused element approach (which failed with error -25212)

2. **Focus Management**: Fixed window activation for menu bar app
   - Temporarily switch from `.accessory` to `.regular` activation policy when showing overlay
   - Call `makeKeyAndOrderFront(nil)` to ensure window gets focus
   - Restore `.accessory` policy when closing overlay (automatically returns focus to original app)
   - Store target app reference when reading text, restore focus before pasting

3. **Apply Flow**: Close overlay before applying expansion
   - Close window first, then apply after 0.1s delay
   - Allows focus to return to original app before pasting
   - Prevents macOS error sound from pasting into wrong window

**Testing Status**: VERIFIED WORKING (2025-12-27)
- Hotkey activation (Cmd+Shift+;)
- Text reading from focused field
- Directive parsing and expansion
- Overlay window shows with focus
- Preview display with warnings
- Apply button replaces text correctly
- Focus returns to original app after apply

**Enhanced Features Added** (2025-12-27):
1. **User Preferences System** (`UserPreferences.swift`)
   - Persistent settings using UserDefaults
   - Skip confirmation toggle
   - Custom snippets path storage
   - Hotkey configuration storage

2. **Enhanced Overlay UI** (`OverlayWindow.swift`)
   - "Don't ask again" checkbox for auto-apply mode
   - "Browse Snippets" button
   - Settings persist immediately when changed

3. **Snippet Browser** (`SnippetBrowserView.swift`, `SnippetInfo.swift`)
   - Search/filter by namespace, description, tags
   - List + detail view
   - Shows operations and template preview
   - TOML parser for snippet metadata extraction

4. **Config Window** (`ConfigWindow.swift`)
   - Shown when hotkey pressed with empty text field
   - Three tabs: Snippets, Settings, About
   - Full snippet browser integration
   - Settings management UI

5. **Auto-Apply Mode** (`AppCoordinator.swift`)
   - When "skip confirmation" enabled, apply immediately
   - No overlay shown for faster workflow
   - Empty text field shows config window instead

**Next Steps**:
1. Test all new features with real usage
2. Add hotkey customization UI
3. Refine snippet library with more domain-specific templates
4. Add Phase 1.6-1.7 features (config management, usage tracking)
5. End-to-end testing with various applications (Phase 3.2)

This plan is updated as implementation progresses and new insights emerge.

