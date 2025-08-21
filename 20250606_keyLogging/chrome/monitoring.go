package chrome

import (
	"encoding/json"
	"log"
	"os"
	"strings"
	"time"

	"keystroke-tracker/metrics"
)

// ChromeEvent represents events from the Chrome extension
type ChromeEvent struct {
	Type       string  `json:"type"`
	FromDomain string  `json:"from_domain,omitempty"`
	ToDomain   string  `json:"to_domain,omitempty"`
	Domain     string  `json:"domain,omitempty"`
	URL        string  `json:"url,omitempty"`
	Title      string  `json:"title,omitempty"`
	Timestamp  float64 `json:"timestamp"`
}

// CurrentDomain holds the currently active Chrome domain
var CurrentDomain string = ""

// Chrome tab session tracking
var currentTabStartTime time.Time
var currentTabDomain string

// MonitorChromeEvents watches Chrome events and updates metrics
func MonitorChromeEvents() {
	chromeEventsFile := "/tmp/keystroke_tracker_chrome_events.jsonl"
	currentDomainFile := "/tmp/keystroke_tracker_current_domain.json"
	
	// For fallback: Chrome profile directory for reading extension storage
	chromeProfileDir := os.Getenv("HOME") + "/Library/Application Support/Google/Chrome/Default"
	
	ticker := time.NewTicker(200 * time.Millisecond) // Check frequently for responsive tab switching
	defer ticker.Stop()

	var lastFileSize int64 = 0
	var lastStorageCheck time.Time

	log.Println("🌐 Chrome monitoring started - checking files and extension storage")

	for {
		select {
		case <-ticker.C:
			// Method 1: Try to read from files (native messaging or direct file write)
			foundFromFiles := false
			
			// Read current domain from file
			if data, err := os.ReadFile(currentDomainFile); err == nil {
				var domainEvent ChromeEvent
				if err := json.Unmarshal(data, &domainEvent); err == nil {
					newDomain := SanitizeDomain(domainEvent.Domain)
					
					// Update session tracking if domain changed
					if CurrentDomain != newDomain {
						updateTabSession(newDomain)
					}
					
					CurrentDomain = newDomain
					foundFromFiles = true
				}
			}

			// Check for new Chrome events from file
			if stat, err := os.Stat(chromeEventsFile); err == nil {
				if stat.Size() > lastFileSize {
					// File has grown, read new content
					if file, err := os.Open(chromeEventsFile); err == nil {
						file.Seek(lastFileSize, 0)
						newData := make([]byte, stat.Size()-lastFileSize)
						file.Read(newData)
						file.Close()
						lines := strings.Split(string(newData), "\n")
						for _, line := range lines {
							if strings.TrimSpace(line) == "" {
								continue
							}

							var event ChromeEvent
							if err := json.Unmarshal([]byte(line), &event); err == nil {
								handleChromeEvent(event)
								foundFromFiles = true
							}
						}
					}
					lastFileSize = stat.Size()
				}
			}
			
			// Method 2: Fallback to Chrome storage if no files found and enough time passed
			if !foundFromFiles && time.Since(lastStorageCheck) > 1*time.Second {
				checkChromeStorage(chromeProfileDir)
				lastStorageCheck = time.Now()
			}
		}
	}
}

// updateTabSession handles Chrome tab session tracking
func updateTabSession(newDomain string) {
	now := time.Now()

	// Record the previous session if we had one
	if currentTabDomain != "" && !currentTabStartTime.IsZero() {
		sessionDuration := now.Sub(currentTabStartTime).Seconds()

		// Get category for metrics (low cardinality)
		currentCategory := GetDomainCategory(currentTabDomain)

		// Record session duration metrics using categories
		metrics.ChromeTabSessionDuration.WithLabelValues(currentCategory).Observe(sessionDuration)
		metrics.ChromeTabTotalTime.WithLabelValues(currentCategory).Add(sessionDuration)

		// Clear the previous tab's gauge
		metrics.CurrentChromeTabGauge.WithLabelValues(currentCategory).Set(0)

		log.Printf("🕒 Chrome session ended: %s [%s] (%.1fs)", currentTabDomain, currentCategory, sessionDuration)
	}

	// Start new session
	currentTabDomain = newDomain
	currentTabStartTime = now

	if newDomain != "" {
		newCategory := GetDomainCategory(newDomain)
		log.Printf("🌐 Chrome session started: %s [%s]", newDomain, newCategory)
	}
}

// UpdateCurrentTabGauge updates the current Chrome tab session duration gauge
func UpdateCurrentTabGauge() {
	if currentTabDomain != "" && !currentTabStartTime.IsZero() {
		sessionDuration := time.Since(currentTabStartTime).Seconds()
		currentCategory := GetDomainCategory(currentTabDomain)
		metrics.CurrentChromeTabGauge.WithLabelValues(currentCategory).Set(sessionDuration)
	}
}

// GetDomainCategory returns domain category for metrics (low cardinality) - exported for use by other packages
func GetDomainCategory(domain string) string {
	domain = strings.ToLower(domain)

	// Development & Code
	if strings.Contains(domain, "github") || strings.Contains(domain, "gitlab") ||
	   strings.Contains(domain, "stackoverflow") || strings.Contains(domain, "codepen") {
		return "development"
	}

	// Google Services
	if strings.Contains(domain, "google") || strings.Contains(domain, "gmail") ||
	   strings.Contains(domain, "drive") || strings.Contains(domain, "docs") {
		return "google_services"
	}

	// Social Media
	if strings.Contains(domain, "twitter") || strings.Contains(domain, "facebook") ||
	   strings.Contains(domain, "linkedin") || strings.Contains(domain, "reddit") {
		return "social_media"
	}

	// Entertainment
	if strings.Contains(domain, "youtube") || strings.Contains(domain, "netflix") ||
	   strings.Contains(domain, "spotify") || strings.Contains(domain, "twitch") {
		return "entertainment"
	}

	// Communication
	if strings.Contains(domain, "slack") || strings.Contains(domain, "teams") ||
	   strings.Contains(domain, "discord") || strings.Contains(domain, "zoom") {
		return "communication"
	}

	// News & Information
	if strings.Contains(domain, "news") || strings.Contains(domain, "wikipedia") ||
	   strings.Contains(domain, "medium") || strings.Contains(domain, "blog") {
		return "information"
	}

	return "other"
}

// logDomainSwitch logs detailed domain switch for analysis (not metrics)
func logDomainSwitch(fromDomain, toDomain string, timestamp float64) {
	switchEvent := map[string]interface{}{
		"timestamp":    timestamp,
		"from_domain":  fromDomain,
		"to_domain":    toDomain,
		"from_category": GetDomainCategory(fromDomain),
		"to_category":   GetDomainCategory(toDomain),
		"event_type":   "domain_switch",
	}

	// Log to structured file for detailed analysis
	jsonData, _ := json.Marshal(switchEvent)
	logLine := string(jsonData) + "\n"

	// Write to domain switches log file
	file, err := os.OpenFile("/tmp/keystroke_tracker_domain_switches.jsonl",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		defer file.Close()
		file.WriteString(logLine)
	}
}

// handleChromeEvent processes Chrome events and updates metrics
func handleChromeEvent(event ChromeEvent) {
	switch event.Type {
	case "domain_switch":
		fromDomain := SanitizeDomain(event.FromDomain)
		toDomain := SanitizeDomain(event.ToDomain)

		// 1. Log detailed switch for analysis (high cardinality, but not metrics)
		logDomainSwitch(fromDomain, toDomain, event.Timestamp)

		// 2. Update metrics with LOW CARDINALITY categories only
		fromCategory := GetDomainCategory(fromDomain)
		toCategory := GetDomainCategory(toDomain)
		metrics.ChromeTabSwitches.WithLabelValues(fromCategory, toCategory).Inc()

		// 3. Update session tracking
		updateTabSession(toDomain)

		// 4. Console log with both detail and category
		log.Printf("🌐 Chrome: %s → %s [%s → %s]", fromDomain, toDomain, fromCategory, toCategory)
	}
}

// checkChromeStorage reads Chrome extension files from Downloads folder
func checkChromeStorage(chromeProfileDir string) {
	homeDir := os.Getenv("HOME")
	downloadsDir := homeDir + "/Downloads"
	
	// Look for Chrome extension files with pattern keystroke_tracker_chrome_*.json
	if files, err := os.ReadDir(downloadsDir); err == nil {
		var latestFile string
		var latestTime int64
		
		for _, file := range files {
			if strings.HasPrefix(file.Name(), "keystroke_tracker_chrome_") && strings.HasSuffix(file.Name(), ".json") {
				if info, err := file.Info(); err == nil {
					if info.ModTime().Unix() > latestTime {
						latestTime = info.ModTime().Unix()
						latestFile = file.Name()
					}
				}
			}
		}
		
		// Read the latest Chrome extension file
		if latestFile != "" {
			filePath := downloadsDir + "/" + latestFile
			if data, err := os.ReadFile(filePath); err == nil {
				var chromeData struct {
					Domain    string  `json:"domain"`
					Timestamp float64 `json:"timestamp"`
					URL       string  `json:"url"`
					Type      string  `json:"type"`
				}
				
				if err := json.Unmarshal(data, &chromeData); err == nil {
					if chromeData.Domain != "" && chromeData.Domain != "unknown" {
						newDomain := SanitizeDomain(chromeData.Domain)
						if CurrentDomain != newDomain {
							updateTabSession(newDomain)
							CurrentDomain = newDomain
							log.Printf("🌐 Chrome domain from Downloads: %s (file: %s)", newDomain, latestFile)
							
							// Clean up old files (keep only the latest 5)
							cleanupOldChromeFiles(downloadsDir)
						}
					}
				}
			}
		}
	}
}

// cleanupOldChromeFiles removes old Chrome extension files to prevent clutter
func cleanupOldChromeFiles(downloadsDir string) {
	if files, err := os.ReadDir(downloadsDir); err == nil {
		var chromeFiles []os.DirEntry
		
		// Collect all Chrome extension files
		for _, file := range files {
			if strings.HasPrefix(file.Name(), "keystroke_tracker_chrome_") && strings.HasSuffix(file.Name(), ".json") {
				chromeFiles = append(chromeFiles, file)
			}
		}
		
		// If more than 5 files, delete the oldest ones
		if len(chromeFiles) > 5 {
			// Sort by modification time (oldest first)
			for i := 0; i < len(chromeFiles)-5; i++ {
				oldFilePath := downloadsDir + "/" + chromeFiles[i].Name()
				os.Remove(oldFilePath)
				log.Printf("🧹 Cleaned up old Chrome file: %s", chromeFiles[i].Name())
			}
		}
	}
}

// UpdateCurrentDomain updates the current Chrome domain (called from HTTP endpoint)
func UpdateCurrentDomain(domain string) {
	newDomain := SanitizeDomain(domain)
	if CurrentDomain != newDomain {
		updateTabSession(newDomain)
		CurrentDomain = newDomain
	}
}

// GetCurrentDomain returns the current Chrome domain for other packages
func GetCurrentDomain() string {
	return CurrentDomain
}

// SanitizeDomain cleans domain names for Prometheus labels (same as extension)
func SanitizeDomain(domain string) string {
	if domain == "" {
		return "unknown"
	}
	// Remove www prefix
	domain = strings.TrimPrefix(domain, "www.")
	// Clean for Prometheus labels
	domain = strings.ReplaceAll(domain, ".", "_")
	domain = strings.ReplaceAll(domain, "-", "_")
	domain = strings.ToLower(domain)
	return domain
}