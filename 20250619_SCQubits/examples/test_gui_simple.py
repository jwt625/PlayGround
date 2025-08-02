#!/usr/bin/env python3
"""
Simple GUI test to check if X11 forwarding is working
"""

import os
print(f"DISPLAY = {os.environ.get('DISPLAY', 'NOT SET')}")

# Test 1: Basic PySide2 import
try:
    from PySide2.QtWidgets import QApplication, QLabel
    from PySide2.QtCore import QTimer
    print("✅ PySide2 imported successfully")
except ImportError as e:
    print(f"❌ PySide2 import failed: {e}")
    exit(1)

# Test 2: Try to create QApplication
try:
    app = QApplication([])
    print("✅ QApplication created successfully")
except Exception as e:
    print(f"❌ QApplication creation failed: {e}")
    print("This usually means X11 forwarding is not working")
    exit(1)

# Test 3: Try to create and show a simple window
try:
    label = QLabel('Hello from Docker PySide2!')
    label.resize(300, 100)
    label.show()
    print("✅ GUI window created and shown")
    print("You should see a window on your macOS desktop!")
    
    # Auto-close after 3 seconds
    QTimer.singleShot(3000, app.quit)
    app.exec_()
    print("✅ GUI test completed successfully")
    
except Exception as e:
    print(f"❌ GUI window creation failed: {e}")
    exit(1)

print("\n🎉 All GUI tests passed! X11 forwarding is working correctly.")
