import Foundation
import AppKit

// Create a simple app detector that prints the current frontmost app
let timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
    if let frontmostApp = NSWorkspace.shared.frontmostApplication {
        print("SWIFT: Current frontmost app: \(frontmostApp.localizedName ?? "Unknown")")
    } else {
        print("SWIFT: No frontmost app detected")
    }
}

// Keep the app running
RunLoop.main.add(timer, forMode: .common)
RunLoop.main.run()