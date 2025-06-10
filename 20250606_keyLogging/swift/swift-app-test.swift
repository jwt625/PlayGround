#!/usr/bin/env swift

import Foundation
import AppKit

print("=== PURE SWIFT APP DETECTION TEST ===")
print("This will show the current frontmost app every 2 seconds")
print("Switch between apps to test detection")
print("Press Ctrl+C to exit\n")

var lastApp = ""
var switchCount = 0

// Create timer that runs every 2 seconds
Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in
    if let frontmostApp = NSWorkspace.shared.frontmostApplication {
        let appName = frontmostApp.localizedName ?? "Unknown"
        let bundleId = frontmostApp.bundleIdentifier ?? "Unknown"
        let pid = frontmostApp.processIdentifier
        
        if appName != lastApp {
            switchCount += 1
            print("🔄 APP SWITCH #\(switchCount): \(lastApp) → \(appName)")
            lastApp = appName
        }
        
        print("Current frontmost app: \(appName) (PID: \(pid), Bundle: \(bundleId))")
        
        // Also show all active regular apps
        let activeApps = NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular && $0.isActive }
            .map { $0.localizedName ?? "Unknown" }
        
        print("All active apps: \(activeApps.joined(separator: ", "))")
        print("---")
        
    } else {
        print("❌ No frontmost application detected")
    }
}

// Keep the program running
RunLoop.main.run()