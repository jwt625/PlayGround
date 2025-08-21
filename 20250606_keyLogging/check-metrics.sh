#!/bin/bash

# Script to check metric cardinality and health
# Helps monitor if metrics explosion is happening again

set -e

echo "📊 Metrics Health Check"
echo "======================"

# Function to check if app is running
check_app_status() {
    echo ""
    echo "🔍 Checking Application Status..."
    
    # Check if Go app is running
    if pgrep -f "keystroke-tracker" > /dev/null; then
        echo "✅ Go application is running"
    else
        echo "❌ Go application is NOT running"
    fi
    
    # Check if Swift helper is running
    if pgrep -f "tracker.swift" > /dev/null; then
        echo "✅ Swift helper is running"
    else
        echo "❌ Swift helper is NOT running"
    fi
    
    # Check if metrics endpoint is accessible
    if curl -s http://localhost:8080/metrics > /dev/null; then
        echo "✅ Metrics endpoint is accessible"
    else
        echo "❌ Metrics endpoint is NOT accessible"
        return 1
    fi
}

# Function to count total metrics
count_total_metrics() {
    echo ""
    echo "📈 Counting Total Metrics..."
    
    local total_metrics=$(curl -s http://localhost:8080/metrics | grep -v '^#' | grep -v '^$' | wc -l | tr -d ' ')
    echo "Total metric time series: $total_metrics"
    
    if [ "$total_metrics" -gt 10000 ]; then
        echo "⚠️  WARNING: High metric count detected! (>10,000)"
        echo "   This may cause issues with free-tier Prometheus backends"
    elif [ "$total_metrics" -gt 1000 ]; then
        echo "⚠️  CAUTION: Moderate metric count (>1,000)"
        echo "   Monitor for further growth"
    else
        echo "✅ Metric count looks healthy (<1,000)"
    fi
    
    return $total_metrics
}

# Function to analyze metric cardinality by name
analyze_metric_cardinality() {
    echo ""
    echo "🔍 Analyzing Metric Cardinality by Name..."
    echo ""
    
    # Get metrics and count by name
    curl -s http://localhost:8080/metrics | grep -v '^#' | grep -v '^$' | \
    sed 's/{.*//' | sort | uniq -c | sort -nr | head -20 | \
    while read count name; do
        if [ "$count" -gt 100 ]; then
            echo "⚠️  $name: $count time series (HIGH)"
        elif [ "$count" -gt 20 ]; then
            echo "⚠️  $name: $count time series (MEDIUM)"
        else
            echo "✅ $name: $count time series (OK)"
        fi
    done
}

# Function to check for problematic labels
check_problematic_labels() {
    echo ""
    echo "🏷️  Checking for Problematic Labels..."
    echo ""
    
    local metrics_output=$(curl -s http://localhost:8080/metrics)
    
    # Check for domain labels with high cardinality
    local domain_count=$(echo "$metrics_output" | grep -o 'domain="[^"]*"' | sort | uniq | wc -l | tr -d ' ')
    if [ "$domain_count" -gt 20 ]; then
        echo "⚠️  High domain label cardinality: $domain_count unique domains"
        echo "   Top domains:"
        echo "$metrics_output" | grep -o 'domain="[^"]*"' | sort | uniq -c | sort -nr | head -10
    else
        echo "✅ Domain label cardinality OK: $domain_count unique domains"
    fi
    
    # Check for app labels
    local app_count=$(echo "$metrics_output" | grep -o 'app="[^"]*"' | sort | uniq | wc -l | tr -d ' ')
    echo "✅ App label cardinality: $app_count unique apps"
    
    # Check for suspicious timestamp-like labels
    if echo "$metrics_output" | grep -q 'timestamp="'; then
        echo "❌ CRITICAL: Found timestamp labels! This will cause metric explosion"
    else
        echo "✅ No timestamp labels found"
    fi
    
    # Check for URL-like labels
    if echo "$metrics_output" | grep -q 'url="'; then
        echo "❌ CRITICAL: Found URL labels! This will cause metric explosion"
    else
        echo "✅ No URL labels found"
    fi
}

# Function to show current Chrome domain categories
show_chrome_categories() {
    echo ""
    echo "🌐 Current Chrome Domain Categories..."
    echo ""
    
    local metrics_output=$(curl -s http://localhost:8080/metrics)
    
    # Show Chrome keystroke categories
    echo "Chrome keystroke categories:"
    echo "$metrics_output" | grep 'chrome_keystrokes_total' | grep -o 'domain_category="[^"]*"' | sort | uniq | sed 's/domain_category="//; s/"//'
    
    echo ""
    echo "Chrome tab switch categories:"
    echo "$metrics_output" | grep 'chrome_tab_switches_total' | grep -o 'from_category="[^"]*"' | sort | uniq | sed 's/from_category="//; s/"//'
}

# Function to estimate Prometheus storage usage
estimate_storage_usage() {
    echo ""
    echo "💾 Estimating Storage Usage..."
    
    local total_metrics=$1
    local estimated_mb=$((total_metrics * 8 / 1024))  # Rough estimate: 8 bytes per sample
    
    echo "Estimated storage per day: ~${estimated_mb}MB"
    echo "Estimated storage per month: ~$((estimated_mb * 30))MB"
    
    if [ $estimated_mb -gt 100 ]; then
        echo "⚠️  High storage usage estimated"
    else
        echo "✅ Storage usage looks reasonable"
    fi
}

# Function to show recommendations
show_recommendations() {
    local total_metrics=$1
    
    echo ""
    echo "💡 Recommendations"
    echo "=================="
    
    if [ "$total_metrics" -gt 5000 ]; then
        echo ""
        echo "🚨 URGENT: Metric count is very high!"
        echo "   1. Run './clear-metrics.sh' to clean up"
        echo "   2. Check for high-cardinality labels"
        echo "   3. Consider removing domain labels from more metrics"
        echo ""
    elif [ "$total_metrics" -gt 1000 ]; then
        echo ""
        echo "⚠️  Metric count is getting high:"
        echo "   1. Monitor growth trends"
        echo "   2. Consider aggregating more labels"
        echo "   3. Review Chrome domain categories"
        echo ""
    else
        echo ""
        echo "✅ Metrics look healthy!"
        echo "   1. Continue monitoring periodically"
        echo "   2. Watch for new high-cardinality labels"
        echo ""
    fi
    
    echo "📊 Useful commands:"
    echo "   - View metrics: curl http://localhost:8080/metrics"
    echo "   - Clear metrics: ./clear-metrics.sh"
    echo "   - Check this again: ./check-metrics.sh"
}

# Main execution
main() {
    # Check if app is running
    if ! check_app_status; then
        echo ""
        echo "❌ Cannot check metrics - application is not running"
        echo "💡 Run './start.sh' to start the application"
        exit 1
    fi
    
    # Count total metrics
    count_total_metrics
    local total_metrics=$?
    
    # Analyze cardinality
    analyze_metric_cardinality
    
    # Check for problematic labels
    check_problematic_labels
    
    # Show Chrome categories
    show_chrome_categories
    
    # Estimate storage
    estimate_storage_usage $total_metrics
    
    # Show recommendations
    show_recommendations $total_metrics
    
    echo ""
    echo "🎉 Health check complete!"
}

# Run main function
main
