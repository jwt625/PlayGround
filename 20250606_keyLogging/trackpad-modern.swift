#!/usr/bin/env swift

import Foundation
import CoreGraphics
import AppKit

// Trackpad click information structure
struct ClickEvent: Codable {
    let buttonType: String
    let app: String
    let timestamp: Double
    let clickCount: Int
}

// File path for communication with Go
let clickEventsFile = "/tmp/keystroke_tracker_trackpad_events.jsonl"

print("🖱️ Trackpad Click Tracker Started")
print("📝 Writing click events to: \(clickEventsFile)")

// Function to get current frontmost app
func getCurrentApp() -> String {
    if let frontmostApp = NSWorkspace.shared.frontmostApplication {
        return frontmostApp.localizedName ?? "Unknown"
    }
    return "Unknown"
}

// Function to log click events
func logClickEvent(buttonType: String, clickCount: Int) {
    let clickEvent = ClickEvent(
        buttonType: buttonType,
        app: getCurrentApp(),
        timestamp: Date().timeIntervalSince1970,
        clickCount: clickCount
    )
    
    do {
        let jsonData = try JSONEncoder().encode(clickEvent)
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            let line = jsonString + "\n"
            
            // Append to file
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

// Mouse event callback
let callback: CGEventTapCallBack = { (proxy, type, event, refcon) in
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

// Create event tap for all mouse buttons
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
    callback: callback,
    userInfo: nil
)

if let tap = eventTap {
    print("✅ CGEvent tap created")
    
    // Add to run loop
    let runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
    
    // Enable the tap
    CGEvent.tapEnable(tap: tap, enable: true)
    
    print("🎯 Monitoring trackpad/mouse clicks...")
    
    // Handle cleanup on exit
    signal(SIGINT) { _ in
        print("\n🛑 Trackpad Click Tracker Stopping")
        
        // Clean up temp files
        try? FileManager.default.removeItem(atPath: clickEventsFile)
        
        exit(0)
    }
    
    CFRunLoopRun()
} else {
    print("❌ Failed to create CGEvent tap")
    print("💡 Needs Accessibility permissions")
}