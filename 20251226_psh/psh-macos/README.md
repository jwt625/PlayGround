# Psh macOS App

macOS menu bar application for expanding psh directives in any text field.

## Features

- **Global Hotkey**: Press `Cmd+Shift+;` to expand psh directives in focused text
- **Accessibility Integration**: Reads and writes text from any application
- **Live Preview**: Shows expansion preview with warnings before applying
- **Menu Bar App**: Runs in the background, no dock icon

## Building

### Prerequisites

- macOS 13.0 or later
- Xcode 14.0 or later
- Rust toolchain (for building the FFI library)

### Build Steps

1. **Build the Rust FFI library**:
   ```bash
   ./build-rust.sh
   ```

2. **Build the Swift app**:
   ```bash
   swift build
   ```

3. **Run the app**:
   ```bash
   swift run
   ```

## Setup

### 1. Accessibility Permissions

The app requires accessibility permissions to read and write text in other applications.

On first launch, the app will request these permissions. You'll need to:
1. Open **System Settings** → **Privacy & Security** → **Accessibility**
2. Enable permissions for **PshMacOS**
3. Restart the app

### 2. Snippets File

The app looks for snippets in the following locations (in order):

1. `~/.config/psh/snippets.toml` (user config)
2. Bundled `snippets.toml` (app resources)
3. `../snippets.toml` (development mode)

To use custom snippets, create `~/.config/psh/snippets.toml`:

```bash
mkdir -p ~/.config/psh
cp ../snippets.toml ~/.config/psh/
```

## Usage

1. **Launch the app** - A semicolon icon appears in the menu bar
2. **Focus any text field** in any application
3. **Type a psh directive** (e.g., `;;d,ne`)
4. **Press `Cmd+Shift+;`** to trigger expansion
5. **Review the preview** in the overlay window
6. **Press Enter** to apply or **Esc** to cancel

## Architecture

```
┌─────────────────────────────────────────┐
│         macOS Application               │
│  ┌───────────────────────────────────┐  │
│  │  AppCoordinator                   │  │
│  │  - Manages app lifecycle          │  │
│  │  - Coordinates components         │  │
│  └───────────────────────────────────┘  │
│           │         │         │          │
│     ┌─────┘    ┌────┘    └────┐         │
│     ▼          ▼              ▼          │
│  ┌──────┐  ┌──────┐      ┌──────┐       │
│  │Hotkey│  │Access│      │ Psh  │       │
│  │Mgr   │  │Mgr   │      │Engine│       │
│  └──────┘  └──────┘      └──────┘       │
│                              │           │
│                              ▼           │
│                          ┌──────┐        │
│                          │ FFI  │        │
│                          │Bridge│        │
│                          └──────┘        │
└──────────────────────────────┼───────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │  Rust psh-core  │
                    │  (via psh-ffi)  │
                    └─────────────────┘
```

## Components

- **main.swift**: App entry point and menu bar setup
- **AppCoordinator.swift**: Main coordinator managing app flow
- **PshWrapper.swift**: Swift wrapper around C FFI
- **AccessibilityManager.swift**: Reads/writes text via AX API
- **HotkeyManager.swift**: Global hotkey registration (Cmd+Shift+;)
- **OverlayWindow.swift**: SwiftUI preview overlay

## Development

### Running in Debug Mode

```bash
swift run
```

### Building for Release

```bash
swift build -c release
```

The executable will be at `.build/release/PshMacOS`.

### Debugging

Enable verbose logging by setting environment variable:

```bash
RUST_LOG=debug swift run
```

## Troubleshooting

### "Accessibility permissions required"
- Grant permissions in System Settings → Privacy & Security → Accessibility
- Restart the app after granting permissions

### "Snippets file not found"
- Ensure `~/.config/psh/snippets.toml` exists
- Or run from the project directory (development mode)

### "Hotkey not working"
- Check if another app is using `Cmd+Shift+;`
- Try restarting the app
- Check Console.app for error messages

### "Cannot read/write text"
- Some apps don't support accessibility API (e.g., password fields)
- Try a different text field
- Check accessibility permissions

## License

Same as parent project.

