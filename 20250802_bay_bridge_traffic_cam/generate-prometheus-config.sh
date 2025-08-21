#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -E '^[A-Z_]+=.*' .env | xargs)
else
    echo "⚠️  Warning: .env file not found"
    exit 1
fi

# Validate required variables
if [ -z "$PROMETHEUS_USERNAME" ] || [ -z "$PROMETHEUS_PUSH_GATEWAY_URL" ] || [ -z "$PROMETHEUS_API_KEY" ]; then
    echo "❌ Missing required environment variables"
    echo "   PROMETHEUS_USERNAME: ${PROMETHEUS_USERNAME:-missing}"
    echo "   PROMETHEUS_PUSH_GATEWAY_URL: ${PROMETHEUS_PUSH_GATEWAY_URL:-missing}"
    echo "   PROMETHEUS_API_KEY: ${PROMETHEUS_API_KEY:-missing}"
    exit 1
fi

echo "✅ Generating prometheus.yml from template..."
echo "   Username: $PROMETHEUS_USERNAME"
echo "   Endpoint: $PROMETHEUS_PUSH_GATEWAY_URL"

# Generate prometheus.yml from template
sed -e "s|__PROMETHEUS_USERNAME__|$PROMETHEUS_USERNAME|g" \
    -e "s|__PROMETHEUS_PUSH_GATEWAY_URL__|$PROMETHEUS_PUSH_GATEWAY_URL|g" \
    -e "s|__PROMETHEUS_API_KEY__|$PROMETHEUS_API_KEY|g" \
    prometheus.yml.template > prometheus.yml

echo "✅ prometheus.yml generated successfully"
