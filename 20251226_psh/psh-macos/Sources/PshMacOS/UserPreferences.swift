import Foundation

/// User preferences for psh app
public class UserPreferences {
    private let defaults = UserDefaults.standard
    
    // Keys
    private enum Keys {
        static let skipConfirmation = "psh.skipConfirmation"
        static let snippetsPath = "psh.snippetsPath"
        static let hotkeyModifiers = "psh.hotkeyModifiers"
        static let hotkeyKeyCode = "psh.hotkeyKeyCode"
    }
    
    public static let shared = UserPreferences()
    
    private init() {}
    
    // MARK: - Skip Confirmation
    
    /// Whether to skip the confirmation overlay and apply expansions immediately
    public var skipConfirmation: Bool {
        get { defaults.bool(forKey: Keys.skipConfirmation) }
        set { defaults.set(newValue, forKey: Keys.skipConfirmation) }
    }
    
    // MARK: - Snippets Path
    
    /// Custom snippets file path (nil = use default search)
    public var customSnippetsPath: String? {
        get { defaults.string(forKey: Keys.snippetsPath) }
        set { defaults.set(newValue, forKey: Keys.snippetsPath) }
    }
    
    // MARK: - Hotkey Configuration
    
    /// Hotkey modifiers (default: Cmd+Shift)
    public var hotkeyModifiers: UInt32 {
        get {
            let stored = defaults.integer(forKey: Keys.hotkeyModifiers)
            return stored == 0 ? 768 : UInt32(stored) // Default: cmdKey + shiftKey
        }
        set { defaults.set(Int(newValue), forKey: Keys.hotkeyModifiers) }
    }
    
    /// Hotkey key code (default: semicolon = 41)
    public var hotkeyKeyCode: UInt32 {
        get {
            let stored = defaults.integer(forKey: Keys.hotkeyKeyCode)
            return stored == 0 ? 41 : UInt32(stored) // Default: semicolon
        }
        set { defaults.set(Int(newValue), forKey: Keys.hotkeyKeyCode) }
    }
    
    // MARK: - Reset
    
    /// Reset all preferences to defaults
    public func resetToDefaults() {
        defaults.removeObject(forKey: Keys.skipConfirmation)
        defaults.removeObject(forKey: Keys.snippetsPath)
        defaults.removeObject(forKey: Keys.hotkeyModifiers)
        defaults.removeObject(forKey: Keys.hotkeyKeyCode)
    }
}

