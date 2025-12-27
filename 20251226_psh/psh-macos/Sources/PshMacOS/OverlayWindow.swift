import SwiftUI
import Cocoa

/// Window controller for the expansion overlay
public class OverlayWindowController: NSWindowController {
    private let onApply: () -> Void
    private let onCancel: () -> Void
    
    public init(
        original: String,
        expanded: String,
        warnings: [Warning],
        onApply: @escaping () -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.onApply = onApply
        self.onCancel = onCancel
        
        // Create the window
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 400),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )

        window.title = "Psh Expansion Preview"
        window.level = .floating
        window.center()
        window.isReleasedWhenClosed = false
        window.hidesOnDeactivate = false

        // Create SwiftUI view
        let contentView = OverlayView(
            original: original,
            expanded: expanded,
            warnings: warnings,
            onApply: { [weak window] in
                window?.close()
                // Small delay to let window close and focus return
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    onApply()
                }
            },
            onCancel: { [weak window] in
                window?.close()
                onCancel()
            }
        )

        window.contentView = NSHostingView(rootView: contentView)

        super.init(window: window)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

/// SwiftUI view for the overlay
struct OverlayView: View {
    let original: String
    let expanded: String
    let warnings: [Warning]
    let onApply: () -> Void
    let onCancel: () -> Void
    
    @State private var selectedTab = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Warnings section (if any)
            if !warnings.isEmpty {
                warningsSection
            }
            
            // Tab selector
            Picker("", selection: $selectedTab) {
                Text("Preview").tag(0)
                Text("Diff").tag(1)
            }
            .pickerStyle(.segmented)
            .padding()
            
            // Content area
            if selectedTab == 0 {
                previewSection
            } else {
                diffSection
            }
            
            // Action buttons
            HStack(spacing: 12) {
                Button("Cancel (Esc)") {
                    onCancel()
                }
                .keyboardShortcut(.cancelAction)
                
                Spacer()
                
                Button("Apply (⏎)") {
                    onApply()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
            .padding()
        }
        .frame(minWidth: 500, minHeight: 300)
    }
    
    private var warningsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                Text("Warnings")
                    .font(.headline)
            }
            
            ForEach(Array(warnings.enumerated()), id: \.offset) { _, warning in
                HStack(alignment: .top, spacing: 8) {
                    Text("•")
                    Text(warning.description)
                        .font(.system(.body, design: .monospaced))
                }
                .foregroundColor(.orange)
            }
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
        .padding()
    }
    
    private var previewSection: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Expanded Text:")
                        .font(.headline)
                    Text(expanded)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(4)
                }
            }
            .padding()
        }
    }
    
    private var diffSection: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Original:")
                        .font(.headline)
                    Text(original)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(4)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Expanded:")
                        .font(.headline)
                    Text(expanded)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(4)
                }
            }
            .padding()
        }
    }
}

