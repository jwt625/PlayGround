#!/bin/bash

echo "🚀 Starting Bay Bridge Traffic Detection with Prometheus monitoring..."

# Generate Prometheus configuration
echo "📝 Generating Prometheus configuration..."
./generate-prometheus-config.sh

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate Prometheus configuration"
    exit 1
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Docker services"
    exit 1
fi

echo "✅ Services started successfully!"
echo ""
echo "📊 Monitoring endpoints:"
echo "   Prometheus: http://localhost:9090"
echo "   Metrics: http://localhost:9091/metrics (when app is running)"
echo ""
echo "🚗 To start traffic detection:"
echo "   python main.py"
echo ""
echo "🧪 To test metrics:"
echo "   python test_metrics.py"
