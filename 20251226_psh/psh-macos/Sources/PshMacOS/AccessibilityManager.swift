import Cocoa
import ApplicationServices
import Carbon

/// Manages accessibility permissions and text reading/writing
/// Uses a simpler approach: Cmd+A to select all, read from pasteboard, then replace
public class AccessibilityManager {

    // Store the target application to restore focus when writing
    private var targetApp: NSRunningApplication?

    public init() {}

    /// Check if accessibility permissions are granted
    public var hasAccessibilityPermissions: Bool {
        return AXIsProcessTrusted()
    }

    /// Request accessibility permissions (opens System Preferences)
    public func requestAccessibilityPermissions() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
        AXIsProcessTrustedWithOptions(options as CFDictionary)
    }

    /// Read text from the currently focused element by selecting all (Cmd+A) and copying
    /// - Returns: The text content, or nil if unable to read
    public func readFocusedText() -> String? {
        guard hasAccessibilityPermissions else {
            print("No accessibility permissions")
            return nil
        }

        // Remember the current frontmost app so we can restore focus later
        targetApp = NSWorkspace.shared.frontmostApplication

        // Save current pasteboard content
        let pasteboard = NSPasteboard.general
        let savedContent = pasteboard.string(forType: .string)

        // Clear pasteboard
        pasteboard.clearContents()

        // Send Cmd+A to select all text
        selectAll()

        // Small delay to let the selection happen
        Thread.sleep(forTimeInterval: 0.05)

        // Send Cmd+C to copy selected text
        copy()

        // Small delay to let the copy happen
        Thread.sleep(forTimeInterval: 0.05)

        // Read from pasteboard
        let text = pasteboard.string(forType: .string)

        // Restore original pasteboard content
        if let saved = savedContent {
            pasteboard.clearContents()
            pasteboard.setString(saved, forType: .string)
        }

        return text
    }

    /// Write text to the currently focused element by selecting all and pasting
    /// - Parameter text: The text to write
    /// - Returns: true if successful
    @discardableResult
    public func writeFocusedText(_ text: String) -> Bool {
        guard hasAccessibilityPermissions else {
            print("No accessibility permissions")
            return false
        }

        // Restore focus to the target app before pasting
        if let app = targetApp {
            print("Restoring focus to: \(app.localizedName ?? "unknown")")
            app.activate(options: .activateIgnoringOtherApps)

            // Wait for the app to become active (with timeout)
            let startTime = Date()
            while !app.isActive && Date().timeIntervalSince(startTime) < 1.0 {
                Thread.sleep(forTimeInterval: 0.05)
            }

            if !app.isActive {
                print("Warning: Target app did not become active")
            }

            // Additional delay to ensure focus is fully restored
            Thread.sleep(forTimeInterval: 0.15)
        } else {
            print("Warning: No target app stored")
        }

        // Save current pasteboard content
        let pasteboard = NSPasteboard.general
        let savedContent = pasteboard.string(forType: .string)

        // Put new text on pasteboard
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        print("Pasting text of length: \(text.count)")

        // Send Cmd+A to select all
        selectAll()

        // Small delay
        Thread.sleep(forTimeInterval: 0.1)

        // Send Cmd+V to paste
        paste()

        // Small delay
        Thread.sleep(forTimeInterval: 0.1)

        // Restore original pasteboard content
        if let saved = savedContent {
            pasteboard.clearContents()
            pasteboard.setString(saved, forType: .string)
        }

        return true
    }

    // MARK: - Keyboard Event Helpers

    private func selectAll() {
        sendKeyPress(keyCode: 0x00, modifiers: .maskCommand) // Cmd+A
    }

    private func copy() {
        sendKeyPress(keyCode: 0x08, modifiers: .maskCommand) // Cmd+C
    }

    private func paste() {
        sendKeyPress(keyCode: 0x09, modifiers: .maskCommand) // Cmd+V
    }

    private func sendKeyPress(keyCode: CGKeyCode, modifiers: CGEventFlags) {
        guard let keyDownEvent = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: true),
              let keyUpEvent = CGEvent(keyboardEventSource: nil, virtualKey: keyCode, keyDown: false) else {
            print("Failed to create keyboard event")
            return
        }

        keyDownEvent.flags = modifiers
        keyUpEvent.flags = modifiers

        keyDownEvent.post(tap: .cghidEventTap)
        keyUpEvent.post(tap: .cghidEventTap)
    }
}

