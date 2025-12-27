import Foundation
import Cocoa

/// Main coordinator that manages the app's core functionality
public class AppCoordinator {
    private let pshEngine = PshEngine()
    private let accessibilityManager = AccessibilityManager()
    private let hotkeyManager = HotkeyManager()

    private var overlayWindowController: OverlayWindowController?
    private var configWindowController: ConfigWindowController?
    private var snippetsPath: String?

    public init() {}
    
    /// Initialize the app
    public func start() {
        // Check and request accessibility permissions
        if !accessibilityManager.hasAccessibilityPermissions {
            print("Requesting accessibility permissions...")
            accessibilityManager.requestAccessibilityPermissions()
            
            // Show alert
            DispatchQueue.main.async {
                let alert = NSAlert()
                alert.messageText = "Accessibility Permissions Required"
                alert.informativeText = "Psh needs accessibility permissions to read and modify text in other applications. Please grant permissions in System Preferences and restart the app."
                alert.alertStyle = .warning
                alert.addButton(withTitle: "OK")
                alert.runModal()
            }
            return
        }
        
        // Find snippets file
        guard let foundSnippetsPath = findSnippetsFile() else {
            showError("Snippets file not found",
                     "Could not find snippets.toml. Please ensure it exists at ~/.config/psh/snippets.toml")
            return
        }

        self.snippetsPath = foundSnippetsPath

        // Initialize psh engine
        guard pshEngine.initialize(snippetsPath: foundSnippetsPath) else {
            showError("Initialization Failed",
                     "Failed to initialize psh engine. Check snippets.toml for errors.")
            return
        }

        print("Psh engine initialized with snippets from: \(foundSnippetsPath)")
        
        // Register hotkey
        let success = hotkeyManager.registerHotkey { [weak self] in
            self?.handleHotkeyPressed()
        }
        
        if success {
            print("Psh is ready! Press Cmd+Shift+; to expand text.")
        } else {
            showError("Hotkey Registration Failed", 
                     "Failed to register global hotkey. The app may not function correctly.")
        }
    }
    
    /// Handle hotkey press
    private func handleHotkeyPressed() {
        print("Hotkey pressed!")

        // Read focused text
        let originalText = accessibilityManager.readFocusedText()

        // If no text field focused OR text is empty, show config/snippet browser
        if originalText == nil || originalText!.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            print("No text or empty field, showing config window")
            showConfigWindow()
            return
        }

        let text = originalText!
        print("Read text: \(text.prefix(100))...")

        // Expand text
        let result = pshEngine.expand(text)

        switch result {
        case .success(let expandedText, let warnings):
            // Check if text actually changed
            if expandedText == text {
                print("No directives found in text")
                showBriefNotification("No psh directives found")
                return
            }

            // Check if we should skip confirmation
            if UserPreferences.shared.skipConfirmation {
                print("Auto-applying expansion (skip confirmation enabled)")
                applyExpansion(expandedText)
                return
            }

            // Show overlay with preview
            showOverlay(
                original: text,
                expanded: expandedText,
                warnings: warnings
            )

        case .failure(let error):
            print("Expansion failed: \(error)")
            showError("Expansion Failed", error)
        }
    }
    
    /// Find the snippets file
    private func findSnippetsFile() -> String? {
        let fileManager = FileManager.default
        
        // Check user config directory first
        let homeDir = fileManager.homeDirectoryForCurrentUser
        let userConfigPath = homeDir
            .appendingPathComponent(".config")
            .appendingPathComponent("psh")
            .appendingPathComponent("snippets.toml")
            .path
        
        if fileManager.fileExists(atPath: userConfigPath) {
            return userConfigPath
        }
        
        // Check bundle resources
        if let bundlePath = Bundle.main.path(forResource: "snippets", ofType: "toml") {
            return bundlePath
        }
        
        // Check relative to executable (for development)
        let executablePath = Bundle.main.executablePath ?? ""
        let devPath = (executablePath as NSString)
            .deletingLastPathComponent
            .appending("/../../../snippets.toml")
        
        if fileManager.fileExists(atPath: devPath) {
            return (devPath as NSString).standardizingPath
        }
        
        return nil
    }
    
    /// Show overlay window with expansion preview
    private func showOverlay(original: String, expanded: String, warnings: [Warning]) {
        DispatchQueue.main.async { [weak self] in
            // Temporarily change to regular app to allow activation
            NSApp.setActivationPolicy(.regular)

            let controller = OverlayWindowController(
                original: original,
                expanded: expanded,
                warnings: warnings,
                snippetsPath: self?.snippetsPath,
                onApply: { [weak self] in
                    // Restore accessory mode before applying
                    NSApp.setActivationPolicy(.accessory)
                    self?.applyExpansion(expanded)
                },
                onCancel: {
                    print("User cancelled")
                    // Restore accessory mode
                    NSApp.setActivationPolicy(.accessory)
                }
            )

            self?.overlayWindowController = controller
            controller.showWindow(nil)

            // Activate our app and make window key
            NSApp.activate(ignoringOtherApps: true)
            controller.window?.makeKeyAndOrderFront(nil)
        }
    }

    /// Show config window (when no text is present)
    private func showConfigWindow() {
        DispatchQueue.main.async { [weak self] in
            // Temporarily change to regular app to allow activation
            NSApp.setActivationPolicy(.regular)

            let controller = ConfigWindowController(
                snippetsPath: self?.snippetsPath,
                onClose: {
                    print("Config window closed")
                    // Restore accessory mode
                    NSApp.setActivationPolicy(.accessory)
                }
            )

            self?.configWindowController = controller
            controller.showWindow(nil)

            // Activate our app and make window key
            NSApp.activate(ignoringOtherApps: true)
            controller.window?.makeKeyAndOrderFront(nil)
        }
    }
    
    /// Apply the expansion by writing to focused text field
    private func applyExpansion(_ text: String) {
        if accessibilityManager.writeFocusedText(text) {
            print("Successfully applied expansion")
        } else {
            showError("Failed to Apply", "Could not write to the focused text field")
        }
    }

    /// Show error alert
    private func showError(_ title: String, _ message: String) {
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.messageText = title
            alert.informativeText = message
            alert.alertStyle = .critical
            alert.addButton(withTitle: "OK")
            alert.runModal()
        }
    }

    /// Show brief notification (TODO: implement as HUD)
    private func showBriefNotification(_ message: String) {
        print("Notification: \(message)")
        // TODO: Show a brief HUD notification instead of just logging
    }
}

