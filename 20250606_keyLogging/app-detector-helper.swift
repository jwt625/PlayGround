#!/usr/bin/env swift

import Foundation
import AppKit

// App information structure
struct AppInfo: Codable {
    let name: String
    let bundleId: String
    let pid: Int32
    let timestamp: Double
}

// File path for communication with Go
let appInfoFile = "/tmp/keystroke_tracker_current_app.json"
let appSwitchFile = "/tmp/keystroke_tracker_app_switches.jsonl"

print("🚀 App Detector Helper Started")
print("📁 Writing current app to: \(appInfoFile)")
print("📝 Writing app switches to: \(appSwitchFile)")

var lastAppName = ""

// Function to write current app info
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

// Function to log app switches
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

// Create timer for monitoring
Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
    if let frontmostApp = NSWorkspace.shared.frontmostApplication {
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
    print("\n🛑 App Detector Helper Stopping")
    
    // Clean up temp files
    try? FileManager.default.removeItem(atPath: appInfoFile)
    try? FileManager.default.removeItem(atPath: appSwitchFile)
    
    exit(0)
}

// Keep running
RunLoop.main.run()