import SwiftUI

/// View for browsing and searching available snippets
struct SnippetBrowserView: View {
    let snippets: [SnippetInfo]
    let onClose: () -> Void
    
    @State private var searchText = ""
    @State private var selectedSnippet: SnippetInfo?
    
    var filteredSnippets: [SnippetInfo] {
        if searchText.isEmpty {
            return snippets
        }
        return snippets.filter { snippet in
            snippet.namespace.localizedCaseInsensitiveContains(searchText) ||
            snippet.description.localizedCaseInsensitiveContains(searchText) ||
            snippet.tags.contains { $0.localizedCaseInsensitiveContains(searchText) }
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Available Snippets")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                        .font(.title2)
                }
                .buttonStyle(.plain)
            }
            .padding()
            
            // Search bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search snippets...", text: $searchText)
                    .textFieldStyle(.plain)
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(8)
            .padding(.horizontal)
            .padding(.bottom, 8)
            
            Divider()
            
            // Snippet list and detail
            HStack(spacing: 0) {
                // List
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(filteredSnippets) { snippet in
                            SnippetListItem(
                                snippet: snippet,
                                isSelected: selectedSnippet?.id == snippet.id
                            )
                            .contentShape(Rectangle())
                            .onTapGesture {
                                selectedSnippet = snippet
                            }
                        }
                    }
                    .padding(8)
                }
                .frame(width: 250)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                
                Divider()
                
                // Detail
                if let snippet = selectedSnippet {
                    SnippetDetailView(snippet: snippet)
                } else {
                    VStack {
                        Spacer()
                        Text("Select a snippet to view details")
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(maxHeight: .infinity)
            
            Divider()
            
            // Footer
            HStack {
                Text("\(filteredSnippets.count) snippet\(filteredSnippets.count == 1 ? "" : "s")")
                    .foregroundColor(.secondary)
                    .font(.caption)
                Spacer()
                Button("Close") {
                    onClose()
                }
                .keyboardShortcut(.escape, modifiers: [])
            }
            .padding()
        }
        .frame(width: 700, height: 500)
        .onAppear {
            selectedSnippet = filteredSnippets.first
        }
    }
}

/// List item for a snippet
struct SnippetListItem: View {
    let snippet: SnippetInfo
    let isSelected: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(";;\(snippet.namespace)")
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.semibold)
                Spacer()
            }
            if !snippet.description.isEmpty {
                Text(snippet.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(8)
        .background(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
        .cornerRadius(6)
    }
}

/// Detail view for a snippet
struct SnippetDetailView: View {
    let snippet: SnippetInfo
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text(";;\(snippet.namespace)")
                        .font(.system(.title2, design: .monospaced))
                        .fontWeight(.bold)
                    
                    if !snippet.description.isEmpty {
                        Text(snippet.description)
                            .foregroundColor(.secondary)
                    }
                }
                
                // Tags
                if !snippet.tags.isEmpty && !snippet.tags.first!.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Tags")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        HStack {
                            ForEach(snippet.tags, id: \.self) { tag in
                                Text(tag)
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.accentColor.opacity(0.2))
                                    .cornerRadius(4)
                            }
                        }
                    }
                }
                
                // Operations
                if !snippet.operations.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Available Operations")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(snippet.operations.joined(separator: ", "))
                            .font(.system(.body, design: .monospaced))
                    }
                }
                
                // Template
                VStack(alignment: .leading, spacing: 4) {
                    Text("Template")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(snippet.template)
                        .font(.system(.caption, design: .monospaced))
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(6)
                }
            }
            .padding()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

