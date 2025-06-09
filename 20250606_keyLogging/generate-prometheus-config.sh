#!/bin/bash

# Generate prometheus.yml from template with environment variables
# Load environment variables from .env file if it exists

if [ -f .env ]; then
    echo "🔧 Loading environment variables from .env file..."
    export $(grep -E '^[A-Z_]+=.*' .env | xargs)
else
    echo "⚠️  Warning: .env file not found. Please create it with:"
    echo "   USERNAME=your_grafana_username"
    echo "   PROMETHEUS_REMOTE_WRITE_ENDPOINT=https://prometheus-prod-xx-xxx.grafana.net/api/prom/push"
    echo "   GRAFANA_API_TOKEN=your_api_token"
    exit 1
fi

# Check if required variables are set
if [ -z "$USERNAME" ] || [ -z "$PROMETHEUS_REMOTE_WRITE_ENDPOINT" ] || [ -z "$GRAFANA_API_TOKEN" ]; then
    echo "❌ Missing required environment variables in .env file"
    echo "   Required: USERNAME, PROMETHEUS_REMOTE_WRITE_ENDPOINT, GRAFANA_API_TOKEN"
    exit 1
fi

echo "✅ Environment variables loaded successfully"
echo "   USERNAME: $USERNAME"
echo "   ENDPOINT: $PROMETHEUS_REMOTE_WRITE_ENDPOINT"
echo "   TOKEN: ${GRAFANA_API_TOKEN:0:10}..."

# Generate prometheus.yml from template
echo "🔄 Generating prometheus.yml from template..."
sed -e "s|__USERNAME__|$USERNAME|g" \
    -e "s|__PROMETHEUS_REMOTE_WRITE_ENDPOINT__|$PROMETHEUS_REMOTE_WRITE_ENDPOINT|g" \
    -e "s|__GRAFANA_API_TOKEN__|$GRAFANA_API_TOKEN|g" \
    prometheus.yml.template > prometheus.yml

echo "✅ prometheus.yml generated successfully"
echo "🚀 You can now run: docker-compose up -d"