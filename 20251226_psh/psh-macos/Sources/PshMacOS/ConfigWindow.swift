import SwiftUI
import Cocoa

/// Window controller for config and snippet browser (shown when no text is selected)
public class ConfigWindowController: NSWindowController {
    private let onClose: () -> Void
    
    public init(
        snippetsPath: String?,
        onClose: @escaping () -> Void
    ) {
        self.onClose = onClose
        
        // Create the window
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 700, height: 600),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )

        window.title = "Psh Configuration"
        window.level = .floating
        window.center()
        window.isReleasedWhenClosed = false
        window.hidesOnDeactivate = false

        // Create SwiftUI view
        let contentView = ConfigView(
            snippetsPath: snippetsPath,
            onClose: { [weak window] in
                window?.close()
                onClose()
            }
        )

        window.contentView = NSHostingView(rootView: contentView)

        super.init(window: window)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

/// SwiftUI view for config and snippet browser
struct ConfigView: View {
    let snippetsPath: String?
    let onClose: () -> Void
    
    @State private var selectedTab = 0
    @State private var skipConfirmation = UserPreferences.shared.skipConfirmation
    
    var snippets: [SnippetInfo] {
        guard let path = snippetsPath else { return [] }
        return SnippetInfoParser.parseSnippetsFile(at: path)
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Psh Configuration")
                    .font(.title)
                    .fontWeight(.semibold)
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                        .font(.title2)
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.escape, modifiers: [])
            }
            .padding()

            // Tab selector
            Picker("", selection: $selectedTab) {
                Text("Snippets").tag(0)
                Text("Settings").tag(1)
                Text("About").tag(2)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.bottom, 8)

            Divider()

            // Content area
            if selectedTab == 0 {
                snippetsTab
            } else if selectedTab == 1 {
                settingsTab
            } else {
                aboutTab
            }
        }
        .frame(minWidth: 700, minHeight: 600)
    }
    
    private var snippetsTab: some View {
        SnippetBrowserView(snippets: snippets, onClose: {})
    }
    
    private var settingsTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Behavior settings
                GroupBox(label: Text("Behavior").font(.headline)) {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle("Skip confirmation (apply expansions immediately)", isOn: $skipConfirmation)
                            .onChange(of: skipConfirmation) { newValue in
                                UserPreferences.shared.skipConfirmation = newValue
                            }
                        
                        Text("When enabled, psh will apply expansions immediately without showing the preview overlay.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }
                
                // Snippets location
                GroupBox(label: Text("Snippets").font(.headline)) {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Snippets file:")
                                .font(.subheadline)
                            Spacer()
                        }
                        
                        if let path = snippetsPath {
                            Text(path)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                                .textSelection(.enabled)
                        } else {
                            Text("No snippets file found")
                                .foregroundColor(.red)
                        }
                        
                        Text("To customize snippets, edit: ~/.config/psh/snippets.toml")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }
                
                // Hotkey settings
                GroupBox(label: Text("Hotkey").font(.headline)) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Current hotkey: Cmd+Shift+;")
                            .font(.subheadline)
                        
                        Text("Hotkey customization coming soon.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }
                
                Spacer()
            }
            .padding()
        }
    }
    
    private var aboutTab: some View {
        VStack(spacing: 20) {
            Spacer()
            
            Image(systemName: "semicolon.circle.fill")
                .font(.system(size: 64))
                .foregroundColor(.accentColor)
            
            Text("Psh")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("Programmable Snippet Helper")
                .font(.title3)
                .foregroundColor(.secondary)

            Text("(also: Prompt Shell)")
                .font(.caption)
                .foregroundColor(.secondary)
                .italic()
            
            Text("Version 0.1.0")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Divider()
                .padding(.horizontal, 100)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Usage:")
                    .font(.headline)
                
                Text("1. Type a psh directive in any text field (e.g., ;;d,ne,l2)")
                    .font(.body)
                
                Text("2. Press Cmd+Shift+; to expand")
                    .font(.body)
                
                Text("3. Review the preview and press Enter to apply")
                    .font(.body)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(8)
            
            Spacer()
            
            Button("Close") {
                onClose()
            }
            .keyboardShortcut(.escape, modifiers: [])
            .padding(.bottom)
        }
        .padding()
    }
}

