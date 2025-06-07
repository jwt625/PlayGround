package metrics

import "github.com/prometheus/client_golang/prometheus"

var (
	// Keystroke metrics with app awareness
	KeystrokesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "keystrokes_total",
			Help: "Total number of keystrokes recorded by key type and application",
		},
		[]string{"key_type", "app"},
	)

	// App time tracking metrics
	AppSessionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "app_session_duration_seconds",
			Help: "Duration of app focus sessions",
			Buckets: []float64{
				// Quick switches (< 1 min)
				0.5, 1, 2, 5, 10, 15, 30,
				// Work sessions (1-60 min, 5-min intervals)
				60, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
				// Extended sessions (1-2h, 10-min intervals)
				4200, 4800, 5400, 6000, 6600, 7200,
				// Long sessions (2-4h, 20-min intervals)
				8400, 9600, 10800, 12000, 13200, 14400,
			},
		},
		[]string{"app"},
	)

	AppTotalTime = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "app_total_time_seconds",
			Help: "Total time spent in each application",
		},
		[]string{"app"},
	)

	CurrentAppGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "current_app_session_seconds",
			Help: "Current session duration for the active app",
		},
		[]string{"app"},
	)

	AppSwitchTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "app_switch_total",
			Help: "Total number of application switches",
		},
	)

	AppSwitchEvents = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "app_switch_events_total",
			Help: "App switch events with from/to app labels",
		},
		[]string{"from_app", "to_app"},
	)

	// Keystroke interval activity - actual counts per 1s window
	KeystrokeIntervalActivity = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "keystroke_interval_activity",
			Help: "Number of keystrokes in the current 1-second interval by app and type",
		},
		[]string{"app", "key_type"},
	)
)

// RegisterMetrics registers all metrics with Prometheus
func RegisterMetrics() {
	prometheus.MustRegister(KeystrokesTotal)
	prometheus.MustRegister(AppSessionDuration)
	prometheus.MustRegister(AppTotalTime)
	prometheus.MustRegister(CurrentAppGauge)
	prometheus.MustRegister(AppSwitchTotal)
	prometheus.MustRegister(AppSwitchEvents)
	prometheus.MustRegister(KeystrokeIntervalActivity)
}