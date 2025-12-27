import Cocoa
import Carbon

/// Manages global hotkey registration and handling
public class HotkeyManager {
    private var eventHandler: EventHandlerRef?
    private var hotKeyRef: EventHotKeyRef?
    private var callback: (() -> Void)?
    
    // Default: Cmd+Shift+; (semicolon)
    private let defaultKeyCode: UInt32 = 41  // ; key
    private let defaultModifiers: UInt32 = UInt32(cmdKey | shiftKey)
    
    public init() {}
    
    /// Register the global hotkey
    /// - Parameter callback: Closure to call when hotkey is pressed
    /// - Returns: true if registration succeeded
    public func registerHotkey(callback: @escaping () -> Void) -> Bool {
        self.callback = callback
        
        // Create event type
        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        
        // Install event handler
        let handler: EventHandlerUPP = { (nextHandler, event, userData) -> OSStatus in
            guard let userData = userData else { return OSStatus(eventNotHandledErr) }
            let manager = Unmanaged<HotkeyManager>.fromOpaque(userData).takeUnretainedValue()
            manager.callback?()
            return noErr
        }
        
        let status = InstallEventHandler(
            GetApplicationEventTarget(),
            handler,
            1,
            &eventType,
            Unmanaged.passUnretained(self).toOpaque(),
            &eventHandler
        )
        
        guard status == noErr else {
            print("Failed to install event handler: \(status)")
            return false
        }
        
        // Register hotkey
        let hotKeyID = EventHotKeyID(signature: OSType(0x5053482E), id: 1) // 'PSH.'
        var hotKeyRefTemp: EventHotKeyRef?
        
        let registerStatus = RegisterEventHotKey(
            defaultKeyCode,
            defaultModifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRefTemp
        )
        
        guard registerStatus == noErr else {
            print("Failed to register hotkey: \(registerStatus)")
            return false
        }
        
        hotKeyRef = hotKeyRefTemp
        print("Hotkey registered: Cmd+Shift+;")
        return true
    }
    
    /// Unregister the global hotkey
    public func unregisterHotkey() {
        if let hotKeyRef = hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
            self.hotKeyRef = nil
        }
        
        if let eventHandler = eventHandler {
            RemoveEventHandler(eventHandler)
            self.eventHandler = nil
        }
        
        callback = nil
    }
    
    deinit {
        unregisterHotkey()
    }
}

