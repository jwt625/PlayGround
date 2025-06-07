package types

import "time"

// AppInfo structure matching Swift helper
type AppInfo struct {
	Name      string  `json:"name"`
	BundleID  string  `json:"bundleId"`
	PID       int32   `json:"pid"`
	Timestamp float64 `json:"timestamp"`
}

// AppSession tracks app focus sessions
type AppSession struct {
	AppName   string
	StartTime time.Time
}

// CounterState for persistence across app restarts
type CounterState struct {
	Keystrokes    map[string]map[string]float64 `json:"keystrokes"`    // [app][key_type]
	AppSwitches   map[string]map[string]float64 `json:"app_switches"`  // [from_app][to_app]
	TotalSwitches float64                       `json:"total_switches"`
}