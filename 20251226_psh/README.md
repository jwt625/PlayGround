# Psh - Programmable Snippet Helper

Text expansion tool with directive-based syntax for dynamic snippet expansion.

## Quick Start

### Build and Run

```bash
# Build Rust libraries
cd psh-macos
./build-rust.sh

# Build and run macOS app
swift build
swift run
```

### Setup

1. **Grant accessibility permissions**: System Settings → Privacy & Security → Accessibility → Enable PshMacOS
2. **Copy snippets**: `mkdir -p ~/.config/psh && cp snippets.toml ~/.config/psh/`
3. **Restart the app**

### Usage

**Normal Mode:**
1. Focus any text field
2. Type a directive: `;;d,ne`
3. Press `Cmd+Shift+;`
4. Review preview, press Enter to apply

**Quick Mode (Auto-Apply):**
1. Enable "Don't ask again" in overlay or settings
2. Type directive and press `Cmd+Shift+;`
3. Text is replaced immediately

**Browse Snippets:**
1. Focus empty text field (or any field)
2. Press `Cmd+Shift+;`
3. Browse available snippets and settings

## Directive Syntax

```
;;namespace,operation,key=value
```

Examples:
- `;;d` → `2025-12-27` (ISO date)
- `;;d,us` → `12/27/2025` (US format)
- `;;d,ne,name=Alice` → `Alice, 2025-12-27`
- `;;e,hi` → Professional greeting email
- `;;c,fn` → Function template

## Architecture

```
psh-core/       Rust engine (parser, resolver, expander)
psh-ffi/        C FFI bindings for Swift
psh-macos/      macOS menu bar app (Swift + SwiftUI)
snippets.toml   Snippet library
```

### Components

**Rust Core**:
- Parser: Parses `;;namespace,op,key=value` syntax
- Snippet: Loads TOML with hierarchical operations
- Resolver: Resolves directives to templates
- Expander: Expands text with variable substitution

**FFI Layer**:
- C-compatible types and functions
- Memory-safe wrapper with mutex protection
- Functions: `psh_init`, `psh_expand`, `psh_free_result`, `psh_reload_snippets`, `psh_shutdown`

**macOS App**:
- `PshWrapper.swift`: Swift FFI wrapper
- `AccessibilityManager.swift`: Read/write text via AX API
- `HotkeyManager.swift`: Global hotkey (Cmd+Shift+;)
- `AppCoordinator.swift`: Main coordinator
- `OverlayWindow.swift`: SwiftUI preview overlay with config options
- `ConfigWindow.swift`: Config and snippet browser window
- `SnippetBrowserView.swift`: Snippet browser UI
- `SnippetInfo.swift`: Snippet metadata parser
- `UserPreferences.swift`: User settings persistence
- `main.swift`: Menu bar app entry point

## Test Results

```
psh-core:  26 tests passing
psh-ffi:    8 tests passing
Total:     35 tests passing
Clippy:     0 warnings
Build:      Success
```

## Customizing Snippets

Edit `~/.config/psh/snippets.toml`:

```toml
[[snippet]]
id = "greeting"
namespace = "g"
template = "Hello {{ name }}!"

[snippet.ops.base]
name = "World"

[snippet.ops.custom]
name = "Custom"
```

Usage: `;;g` → `Hello World!`, `;;g,custom` → `Hello Custom!`

## Troubleshooting

**"Accessibility permissions required"**: Grant in System Settings, restart app

**"No psh directives found"**: Ensure `;;` prefix and valid namespace

**"Snippets file not found"**: Copy to `~/.config/psh/snippets.toml`

**Hotkey not working**: Check for conflicts with other apps

## Development

### Build Rust Library

```bash
cd psh-ffi
cargo build --release
```

### Run Tests

```bash
cargo test --workspace
cargo test --workspace -- --nocapture test_comprehensive_examples
cargo clippy --workspace -- -D warnings
```

### Build macOS App

```bash
cd psh-macos
swift build -c release
```

## Documentation

- `DevLogs/DevLog-001-implementation_plan.md`: Design decisions and implementation plan
- `psh-macos/README.md`: macOS app architecture details
- `psh-ffi/psh.h`: C API documentation

## Features

- [x] Directive parsing and expansion
- [x] Global hotkey (Cmd+Shift+;)
- [x] Preview overlay with diff view
- [x] Snippet browser with search
- [x] Auto-apply mode (skip confirmation)
- [x] User preferences persistence
- [x] Config window (shown when no text present)
- [x] Warning display for unknown directives
- [ ] Hotkey customization UI
- [ ] Usage statistics tracking

## Status

Phase 1 (Rust Core): Complete
Phase 2 (macOS App): Complete with enhanced UI
Ready for testing and feedback.

See `DevLogs/DevLog-001-implementation_plan.md` for detailed implementation status.

