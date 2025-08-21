#!/bin/bash

echo "=== Bay Bridge Traffic - Cloudflare Tunnel Setup ==="
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared is not installed. Please run: brew install cloudflared"
    exit 1
fi

echo "✅ cloudflared is installed"
echo ""

# Step 1: Login to Cloudflare
echo "Step 1: Authenticate with Cloudflare"
echo "This will open a browser window for you to login to Cloudflare..."
read -p "Press Enter to continue..."

cloudflared tunnel login

if [ $? -ne 0 ]; then
    echo "❌ Failed to authenticate with Cloudflare"
    exit 1
fi

echo "✅ Successfully authenticated with Cloudflare"
echo ""

# Step 2: Create tunnel
echo "Step 2: Creating tunnel 'grafana-local'"
cloudflared tunnel create grafana-local

if [ $? -ne 0 ]; then
    echo "❌ Failed to create tunnel. It might already exist."
    echo "You can list existing tunnels with: cloudflared tunnel list"
fi

echo "✅ Tunnel 'grafana-local' created"
echo ""

# Step 3: Create config directory
echo "Step 3: Setting up tunnel configuration"
mkdir -p ~/.cloudflared

# Step 4: Create tunnel config
# Get the tunnel ID to construct the correct credentials file path
TUNNEL_ID=$(cloudflared tunnel list | grep grafana-local | awk '{print $1}')
if [ -z "$TUNNEL_ID" ]; then
    echo "❌ Could not find tunnel ID for grafana-local"
    exit 1
fi

cat > ~/.cloudflared/config.yml << EOF
tunnel: grafana-local
credentials-file: ~/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: bay-bridge-traffic.com
    service: http://localhost:8080
  - service: http_status:404
EOF

echo "✅ Tunnel configuration created at ~/.cloudflared/config.yml"
echo ""

# Step 5: Route DNS
echo "Step 5: Setting up DNS routing"
echo "This will point bay-bridge-traffic.com to your tunnel..."
cloudflared tunnel route dns grafana-local bay-bridge-traffic.com

if [ $? -ne 0 ]; then
    echo "❌ Failed to set up DNS routing"
    echo "You may need to manually add the DNS record in Cloudflare dashboard"
else
    echo "✅ DNS routing configured"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start the tunnel, run:"
echo "  cloudflared tunnel run grafana-local"
echo ""
echo "Your Grafana dashboard will be available at:"
echo "  https://bay-bridge-traffic.com"
echo ""
echo "Local services:"
echo "  - Grafana: http://localhost:3000"
echo "  - Nginx Proxy: http://localhost:8080"
echo "  - Prometheus: http://localhost:9090"
echo "  - Metrics Server: http://localhost:9091"
