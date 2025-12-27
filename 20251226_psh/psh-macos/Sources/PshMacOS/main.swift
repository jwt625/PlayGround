import Cocoa

// Create app delegate
class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var coordinator: AppCoordinator?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create menu bar item
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem?.button {
            button.image = NSImage(systemSymbolName: "semicolon.circle", accessibilityDescription: "Psh")
        }

        // Create menu
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "About Psh", action: #selector(showAbout), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        statusItem?.menu = menu

        // Start coordinator
        coordinator = AppCoordinator()
        coordinator?.start()
    }

    @objc func showAbout() {
        let alert = NSAlert()
        alert.messageText = "Psh - Programmable Snippet Helper"
        alert.informativeText = """
        Version 0.1.0

        Press Cmd+Shift+; to expand psh directives in any text field.

        Snippets location: ~/.config/psh/snippets.toml
        """
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}

// Run the app
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory) // Menu bar app, no dock icon
app.run()

print("Hello, world!")
