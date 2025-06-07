package app

import (
	"encoding/json"
	"os"
	"strings"
	"time"
)

// MonitorAppSwitches watches the app switch log file for changes
func MonitorAppSwitches() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastFileSize int64 = 0

	for {
		select {
		case <-ticker.C:
			// Check for new app switches in the log file
			if stat, err := os.Stat("/tmp/keystroke_tracker_app_switches.jsonl"); err == nil {
				if stat.Size() > lastFileSize {
					// File has grown, read new content
					if file, err := os.Open("/tmp/keystroke_tracker_app_switches.jsonl"); err == nil {
						file.Seek(lastFileSize, 0) // Start reading from where we left off
						newData := make([]byte, stat.Size()-lastFileSize)
						file.Read(newData)
						file.Close()
						lines := strings.Split(string(newData), "\n")
						for _, line := range lines {
							if strings.TrimSpace(line) == "" {
								continue
							}

							var switchInfo map[string]interface{}
							if err := json.Unmarshal([]byte(line), &switchInfo); err == nil {
								fromApp := SanitizeAppName(switchInfo["from"].(string))
								toApp := SanitizeAppName(switchInfo["to"].(string))
								timestamp := switchInfo["timestamp"].(float64)

								// Handle app session tracking
								HandleAppSwitch(fromApp, toApp, time.Unix(int64(timestamp), 0))
							}
						}
					}
					lastFileSize = stat.Size()
				}
			}
		}
	}
}