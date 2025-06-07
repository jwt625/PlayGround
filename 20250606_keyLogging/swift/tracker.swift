#!/usr/bin/env swift

import Foundation
import CoreGraphics
import AppKit

// Data structures
struct AppInfo: Codable {
    let name: String
    let bundleId: String
    let pid: Int32
    let timestamp: Double
}

struct ClickEvent: Codable {
    let buttonType: String
    let app: String
    let timestamp: Double
    let clickCount: Int
}

// File paths for communication with Go
let appInfoFile = "/tmp/keystroke_tracker_current_app.json"
let appSwitchFile = "/tmp/keystroke_tracker_app_switches.jsonl"
let clickEventsFile = "/tmp/keystroke_tracker_trackpad_events.jsonl"

print("🚀 Unified Tracker Started")
print("📱 App detection + 🖱️ Trackpad clicks")
print("📁 Current app: \(appInfoFile)")
print("📝 App switches: \(appSwitchFile)")
print("📝 Click events: \(clickEventsFile)")

var lastAppName = ""

// Shared function to get current frontmost app
func getCurrentApp() -> NSRunningApplication? {
    return NSWorkspace.shared.frontmostApplication
}

func getCurrentAppName() -> String {
    return getCurrentApp()?.localizedName ?? "Unknown"
}

// App detection functions
func writeCurrentApp(_ app: NSRunningApplication) {
    let appInfo = AppInfo(
        name: app.localizedName ?? "Unknown",
        bundleId: app.bundleIdentifier ?? "unknown",
        pid: app.processIdentifier,
        timestamp: Date().timeIntervalSince1970
    )
    
    do {
        let jsonData = try JSONEncoder().encode(appInfo)
        try jsonData.write(to: URL(fileURLWithPath: appInfoFile))
    } catch {
        print("❌ Error writing app info: \(error)")
    }
}

func logAppSwitch(from: String, to: String, timestamp: Double) {
    let switchInfo: [String: Any] = [
        "from": from,
        "to": to,
        "timestamp": timestamp
    ]
    
    do {
        let jsonData = try JSONSerialization.data(withJSONObject: switchInfo)
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            let line = jsonString + "\n"
            
            if FileManager.default.fileExists(atPath: appSwitchFile) {
                if let fileHandle = FileHandle(forWritingAtPath: appSwitchFile) {
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(line.data(using: .utf8)!)
                    fileHandle.closeFile()
                }
            } else {
                try line.write(toFile: appSwitchFile, atomically: true, encoding: .utf8)
            }
        }
    } catch {
        print("❌ Error logging app switch: \(error)")
    }
}

// Trackpad click functions
func logClickEvent(buttonType: String, clickCount: Int) {
    let clickEvent = ClickEvent(
        buttonType: buttonType,
        app: getCurrentAppName(),
        timestamp: Date().timeIntervalSince1970,
        clickCount: clickCount
    )
    
    do {
        let jsonData = try JSONEncoder().encode(clickEvent)
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            let line = jsonString + "\n"
            
            if FileManager.default.fileExists(atPath: clickEventsFile) {
                if let fileHandle = FileHandle(forWritingAtPath: clickEventsFile) {
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(line.data(using: .utf8)!)
                    fileHandle.closeFile()
                }
            } else {
                try line.write(toFile: clickEventsFile, atomically: true, encoding: .utf8)
            }
            
            print("🖱️ \(buttonType.capitalized) click in \(clickEvent.app) (count: \(clickCount))")
        }
    } catch {
        print("❌ Error logging click: \(error)")
    }
}

// Mouse event callback for trackpad clicks
let mouseCallback: CGEventTapCallBack = { (proxy, type, event, refcon) in
    let buttonType: String
    switch type {
    case .leftMouseDown:
        buttonType = "left"
    case .rightMouseDown:
        buttonType = "right"
    case .otherMouseDown:
        buttonType = "middle"
    default:
        return Unmanaged.passRetained(event)
    }
    
    let clickCount = Int(event.getIntegerValueField(.mouseEventClickState))
    logClickEvent(buttonType: buttonType, clickCount: clickCount)
    
    return Unmanaged.passRetained(event)
}

// Set up trackpad monitoring
let eventMask = CGEventMask(
    (1 << CGEventType.leftMouseDown.rawValue) |
    (1 << CGEventType.rightMouseDown.rawValue) |
    (1 << CGEventType.otherMouseDown.rawValue)
)

let eventTap = CGEvent.tapCreate(
    tap: .cghidEventTap,
    place: .headInsertEventTap,
    options: .listenOnly,
    eventsOfInterest: eventMask,
    callback: mouseCallback,
    userInfo: nil
)

if let tap = eventTap {
    print("✅ Trackpad monitoring started")
    
    let runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
    CGEvent.tapEnable(tap: tap, enable: true)
} else {
    print("❌ Failed to start trackpad monitoring")
    print("💡 Needs Accessibility permissions")
    exit(1)
}

// Set up app monitoring timer
Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
    if let frontmostApp = getCurrentApp() {
        let currentAppName = frontmostApp.localizedName ?? "Unknown"
        
        // Write current app info
        writeCurrentApp(frontmostApp)
        
        // Check for app switch
        if currentAppName != lastAppName && !lastAppName.isEmpty {
            let timestamp = Date().timeIntervalSince1970
            print("🔄 App Switch: \(lastAppName) → \(currentAppName)")
            logAppSwitch(from: lastAppName, to: currentAppName, timestamp: timestamp)
        }
        
        lastAppName = currentAppName
    }
}

// Handle cleanup on exit
signal(SIGINT) { _ in
    print("\n🛑 Unified Tracker Stopping")
    
    // Clean up temp files
    try? FileManager.default.removeItem(atPath: appInfoFile)
    try? FileManager.default.removeItem(atPath: appSwitchFile)
    try? FileManager.default.removeItem(atPath: clickEventsFile)
    
    exit(0)
}

print("🎯 Monitoring apps and trackpad clicks...")
RunLoop.main.run()