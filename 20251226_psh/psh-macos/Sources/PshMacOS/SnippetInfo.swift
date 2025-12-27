import Foundation

/// Represents snippet information for display in the UI
public struct SnippetInfo: Identifiable, Hashable {
    public let id: String
    public let namespace: String
    public let description: String
    public let tags: [String]
    public let operations: [String]
    public let template: String
    
    public init(id: String, namespace: String, description: String, tags: [String], operations: [String], template: String) {
        self.id = id
        self.namespace = namespace
        self.description = description
        self.tags = tags
        self.operations = operations
        self.template = template
    }
}

/// Parser for extracting snippet information from snippets.toml
public class SnippetInfoParser {
    
    /// Parse snippets.toml file and extract snippet information
    public static func parseSnippetsFile(at path: String) -> [SnippetInfo] {
        guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            return []
        }
        
        return parseSnippetsContent(content)
    }
    
    /// Parse snippets content and extract snippet information
    public static func parseSnippetsContent(_ content: String) -> [SnippetInfo] {
        var snippets: [SnippetInfo] = []
        
        // Simple TOML parser for snippet sections
        let lines = content.components(separatedBy: .newlines)
        var currentSnippet: [String: String] = [:]
        var currentOps: [String] = []
        var inSnippet = false
        var inTemplate = false
        var templateLines: [String] = []
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            // Start of snippet
            if trimmed == "[[snippet]]" {
                // Save previous snippet if exists
                if inSnippet, let id = currentSnippet["id"], let namespace = currentSnippet["namespace"] {
                    let snippet = SnippetInfo(
                        id: id,
                        namespace: namespace,
                        description: currentSnippet["description"] ?? "",
                        tags: parseTags(currentSnippet["tags"] ?? ""),
                        operations: currentOps,
                        template: templateLines.joined(separator: "\n")
                    )
                    snippets.append(snippet)
                }
                
                // Reset for new snippet
                currentSnippet = [:]
                currentOps = []
                templateLines = []
                inSnippet = true
                inTemplate = false
                continue
            }
            
            // End of snippet section (start of ops)
            if trimmed.hasPrefix("[snippet.ops.") {
                inTemplate = false
                if let opName = trimmed.split(separator: ".").last?.dropLast().description {
                    currentOps.append(opName)
                }
                continue
            }
            
            // Skip global ops and other sections
            if trimmed.hasPrefix("[global") || trimmed == "[global]" {
                inSnippet = false
                continue
            }
            
            if !inSnippet {
                continue
            }
            
            // Template start
            if trimmed.hasPrefix("template = \"\"\"") {
                inTemplate = true
                continue
            }
            
            // Template end
            if inTemplate && trimmed == "\"\"\"" {
                inTemplate = false
                continue
            }
            
            // Template content
            if inTemplate {
                templateLines.append(line)
                continue
            }
            
            // Parse key-value pairs
            if let equalIndex = trimmed.firstIndex(of: "=") {
                let key = String(trimmed[..<equalIndex]).trimmingCharacters(in: .whitespaces)
                let value = String(trimmed[trimmed.index(after: equalIndex)...])
                    .trimmingCharacters(in: .whitespaces)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                currentSnippet[key] = value
            }
        }
        
        // Save last snippet
        if inSnippet, let id = currentSnippet["id"], let namespace = currentSnippet["namespace"] {
            let snippet = SnippetInfo(
                id: id,
                namespace: namespace,
                description: currentSnippet["description"] ?? "",
                tags: parseTags(currentSnippet["tags"] ?? ""),
                operations: currentOps,
                template: templateLines.joined(separator: "\n")
            )
            snippets.append(snippet)
        }
        
        return snippets
    }
    
    private static func parseTags(_ tagsString: String) -> [String] {
        // Parse ["tag1", "tag2"] format
        let cleaned = tagsString
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .replacingOccurrences(of: "\"", with: "")
        return cleaned.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }
}

