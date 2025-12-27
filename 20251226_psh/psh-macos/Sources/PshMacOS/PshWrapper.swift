import Foundation
import PshFFI

/// Swift wrapper for psh FFI
/// Provides memory-safe Swift API over C FFI functions
public class PshEngine {
    private var isInitialized = false
    
    public init() {}
    
    /// Initialize the psh engine with snippets from a TOML file
    /// - Parameter snippetsPath: Path to the snippets.toml file
    /// - Returns: true if initialization succeeded
    public func initialize(snippetsPath: String) -> Bool {
        let result = snippetsPath.withCString { path in
            psh_init(path)
        }
        isInitialized = result
        return result
    }
    
    /// Expand text containing psh directives
    /// - Parameter text: Input text with psh directives
    /// - Returns: Result containing expanded text and warnings, or error
    public func expand(_ text: String) -> ExpansionResult {
        guard isInitialized else {
            return .failure("Engine not initialized. Call initialize() first.")
        }
        
        return text.withCString { cText in
            guard let resultPtr = psh_expand(cText) else {
                return .failure("Expansion failed: null result")
            }
            
            defer { psh_free_result(resultPtr) }
            
            let result = resultPtr.pointee
            
            // Check for error first
            if let errorPtr = result.error {
                let errorMsg = String(cString: errorPtr)
                return .failure(errorMsg)
            }
            
            // Extract expanded text
            guard let textPtr = result.text else {
                return .failure("Expansion failed: no text returned")
            }
            let expandedText = String(cString: textPtr)
            
            // Extract warnings
            var warnings: [Warning] = []
            if result.warning_count > 0, let warningsPtr = result.warnings {
                for i in 0..<Int(result.warning_count) {
                    let warning = warningsPtr[i]
                    let warningType = WarningType(rawValue: warning.warning_type) ?? .unknownNamespace
                    let namespace = warning.namespace != nil ? String(cString: warning.namespace) : ""
                    let name = warning.name != nil ? String(cString: warning.name) : ""
                    warnings.append(Warning(type: warningType, namespace: namespace, name: name))
                }
            }
            
            return .success(expandedText, warnings)
        }
    }
    
    /// Reload snippets from a file
    /// - Parameter snippetsPath: Path to the snippets.toml file
    /// - Returns: true if reload succeeded
    public func reloadSnippets(snippetsPath: String) -> Bool {
        return snippetsPath.withCString { path in
            psh_reload_snippets(path)
        }
    }
    
    /// Shutdown the psh engine and free all resources
    public func shutdown() {
        psh_shutdown()
        isInitialized = false
    }
    
    deinit {
        if isInitialized {
            shutdown()
        }
    }
}

/// Result of text expansion
public enum ExpansionResult {
    case success(String, [Warning])
    case failure(String)
    
    public var text: String? {
        if case .success(let text, _) = self {
            return text
        }
        return nil
    }
    
    public var warnings: [Warning] {
        if case .success(_, let warnings) = self {
            return warnings
        }
        return []
    }
    
    public var error: String? {
        if case .failure(let error) = self {
            return error
        }
        return nil
    }
    
    public var isSuccess: Bool {
        if case .success = self {
            return true
        }
        return false
    }
}

/// Warning type enumeration
public enum WarningType: UInt32 {
    case unknownNamespace = 0
    case unknownOperation = 1
    case unknownKey = 2
    
    public var description: String {
        switch self {
        case .unknownNamespace: return "Unknown Namespace"
        case .unknownOperation: return "Unknown Operation"
        case .unknownKey: return "Unknown Key"
        }
    }
}

/// Warning structure
public struct Warning {
    public let type: WarningType
    public let namespace: String
    public let name: String
    
    public var description: String {
        switch type {
        case .unknownNamespace:
            return "Unknown namespace: '\(namespace)'"
        case .unknownOperation:
            return "Unknown operation '\(name)' in namespace '\(namespace)'"
        case .unknownKey:
            return "Unknown key '\(name)' in namespace '\(namespace)'"
        }
    }
}

