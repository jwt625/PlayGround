#!/bin/bash
# Build ScreenCap.app (menu bar region recorder with system audio) into ./build, ad-hoc signed.
# Optional: ./build.sh install  -> also copies to ~/Applications/ScreenCap.app
set -euo pipefail
cd "$(dirname "$0")"

APP=build/ScreenCap.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

swiftc -O -swift-version 5 -target arm64-apple-macos13.0 \
  -framework AppKit -framework ScreenCaptureKit -framework AVFoundation -framework AVKit -framework Carbon \
  src/main.swift -o "$APP/Contents/MacOS/ScreenCap"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>io.github.jwt625.screencap</string>
  <key>CFBundleName</key><string>ScreenCap</string>
  <key>CFBundleExecutable</key><string>ScreenCap</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

codesign --force --sign - "$APP"
echo "built $APP"

if [[ "${1:-}" == "install" ]]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/ScreenCap.app"
  cp -R "$APP" "$HOME/Applications/ScreenCap.app"
  echo "installed ~/Applications/ScreenCap.app"
fi
