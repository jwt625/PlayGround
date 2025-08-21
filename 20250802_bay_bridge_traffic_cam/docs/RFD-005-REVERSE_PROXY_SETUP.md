# RFD-005: Bay Bridge Traffic Dashboard Reverse Proxy Setup

## Overview

This RFD describes the implementation of a reverse proxy solution to host the Bay Bridge Traffic Detection dashboard at a custom domain (https://bay-bridge-traffic.com) without redirecting to Grafana Cloud. The solution uses a local Grafana instance, Nginx reverse proxy, and Cloudflare Tunnel to provide secure public access.

## Problem Statement

The existing setup uses Grafana Cloud's public dashboard feature, but this has limitations:
- Public dashboards cannot access private data sources for security reasons
- Limited customization and interaction capabilities
- URLs redirect to jwt625.grafana.net domain
- No control over the user experience

## Solution Architecture

```
Traffic Detection App (port 9091)
         ↓
Local Prometheus (port 9090) ──remote_write──→ Grafana Cloud Prometheus
         ↑                                              ↑
         │                                              │
    [Existing Flow]                              [New Dashboard Access]
                                                         │
Internet → Cloudflare Tunnel → Nginx (port 8080) → Local Grafana (port 3000)
```

### Components Overview

1. **Local Grafana Instance**: Hosts the dashboard with full functionality
2. **Nginx Reverse Proxy**: Provides clean URL mapping and SSL termination
3. **Cloudflare Tunnel**: Securely exposes local services to the internet
4. **Remote Data Source**: Connects to Grafana Cloud Prometheus with authentication

## New Components Added

### 1. Local Grafana Instance
- **Purpose**: Host dashboard with full functionality and remote data access
- **Port**: 3000
- **Configuration**: Auto-provisioned with data source and dashboard
- **Data Source**: Grafana Cloud Prometheus with authentication
- **Dashboard**: Imported from existing `grafana-dashboard.json`

### 2. Nginx Reverse Proxy
- **Purpose**: Clean URL mapping and request forwarding
- **Port**: 8080
- **Configuration**: `/opt/homebrew/etc/nginx/servers/bay-bridge-traffic.conf`
- **Features**: WebSocket support, proper headers, timeouts

### 3. Cloudflare Tunnel (cloudflared)
- **Purpose**: Secure internet exposure without port forwarding
- **Tunnel Name**: `grafana-local`
- **Configuration**: `~/.cloudflared/config.yml`
- **DNS**: Automatic CNAME record creation

### 4. Enhanced Docker Compose
- **Added**: Grafana service with proper environment variables
- **Volumes**: Persistent storage and provisioning directories
- **Networks**: Shared monitoring network

## Updated File Structure

```
20250802_bay_bridge_traffic_cam/
├── docker-compose.yml                    # Updated with Grafana service
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml           # Remote Prometheus config
│   │   └── dashboards/
│   │       └── dashboards.yml           # Dashboard provider config
│   └── dashboards/
│       └── grafana-dashboard.json       # Imported dashboard
├── nginx/
│   └── bay-bridge-traffic.conf          # Nginx virtual host config
├── setup-cloudflare-tunnel.sh           # Automated tunnel setup
├── start-bay-bridge-services.sh         # Service startup script
└── docs/
    └── RFD-005-REVERSE_PROXY_SETUP.md   # This document
```

## Implementation Status

✅ **Local Grafana**: Running on port 3000 with dashboard auto-imported and authentication disabled
✅ **Nginx Reverse Proxy**: Configured and running on port 8080
✅ **Prometheus**: Running on port 9090 with local data source configured
✅ **Metrics Server**: Running on port 9091 (existing, unchanged)
✅ **Cloudflared**: Installed and tunnel created successfully
✅ **Automation Scripts**: Created for easy setup and management
✅ **DNS Routing**: Configured in Cloudflare (manual setup required)
✅ **Static Asset Loading**: CSS and JS files now loading correctly
✅ **Tunnel Connectivity**: Cloudflare tunnel running with 4 active connections
⚠️ **WebSocket Connections**: Grafana Live WebSocket connections failing through reverse proxy
⚠️ **API Endpoints**: Some API endpoints (e.g., `/api/frontend-metrics`) returning 404 through proxy

## Quick Start

### 1. Start All Services
```bash
./start-bay-bridge-services.sh
```

### 2. Set Up Cloudflare Tunnel (One-time setup)
```bash
./setup-cloudflare-tunnel.sh
```

### 3. Start the Tunnel
```bash
cloudflared tunnel run grafana-local
```

### 4. Access Your Dashboard
Visit: https://bay-bridge-traffic.com

## Manual Setup Steps

### Prerequisites
- Docker running
- Domain `bay-bridge-traffic.com` managed by Cloudflare
- Cloudflare account access

### Step-by-Step

1. **Start Local Services**
   ```bash
   # Start Prometheus + Grafana
   docker-compose up -d
   
   # Start Nginx
   nginx
   ```

2. **Authenticate with Cloudflare**
   ```bash
   cloudflared tunnel login
   ```

3. **Create Tunnel**
   ```bash
   cloudflared tunnel create grafana-local
   ```

4. **Configure Tunnel**
   Create `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: grafana-local
   credentials-file: ~/.cloudflared/grafana-local.json
   
   ingress:
     - hostname: bay-bridge-traffic.com
       service: http://localhost:8080
     - service: http_status:404
   ```

5. **Set Up DNS**
   ```bash
   cloudflared tunnel route dns grafana-local bay-bridge-traffic.com
   ```

6. **Run Tunnel**
   ```bash
   cloudflared tunnel run grafana-local
   ```

## Service URLs

- **Public Dashboard**: https://bay-bridge-traffic.com
- **Local Grafana**: http://localhost:3000 (admin/admin)
- **Local Prometheus**: http://localhost:9090
- **Nginx Proxy**: http://localhost:8080
- **Metrics Server**: http://localhost:9091

## Configuration Details

### Docker Compose Updates (`docker-compose.yml`)
```yaml
# Added Grafana service with authentication disabled
grafana:
  image: grafana/grafana:latest
  container_name: grafana
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
    - GF_USERS_ALLOW_SIGN_UP=false
    - GF_SERVER_DOMAIN=bay-bridge-traffic.com
    - GF_SERVER_ROOT_URL=https://bay-bridge-traffic.com/
    - GF_SERVER_SERVE_FROM_SUB_PATH=false
    - GF_AUTH_ANONYMOUS_ENABLED=true
    - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    - GF_AUTH_DISABLE_LOGIN_FORM=true
    - GF_AUTH_DISABLE_SIGNOUT_MENU=true
    # Additional reverse proxy settings
    - GF_SERVER_ENABLE_GZIP=true
    - GF_LOG_LEVEL=debug
  volumes:
    - grafana-storage:/var/lib/grafana
    - ./grafana/provisioning:/etc/grafana/provisioning
    - ./grafana/dashboards:/var/lib/grafana/dashboards
```

### Nginx Configuration (`nginx/bay-bridge-traffic.conf`)
```nginx
server {
    listen 8080;
    server_name _;

    # Redirect root to dashboard
    location = / {
        return 302 /d/bay-bridge-traffic/bay-bridge-traffic-detection-system;
    }

    # Proxy all requests to Grafana
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-Host bay-bridge-traffic.com;
        proxy_set_header X-Forwarded-Port 443;

        # WebSocket support for Grafana live features
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Increase timeouts for long-running queries
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffer settings
        proxy_buffering off;
        proxy_request_buffering off;
    }
}
```

### Grafana Data Source (`grafana/provisioning/datasources/prometheus.yml`)
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://host.docker.internal:9090
    uid: prometheus
    isDefault: true
    editable: false
```

**Note**: Updated to use local Prometheus instance instead of Grafana Cloud to avoid authentication issues. The `host.docker.internal:9090` allows the Grafana container to access the local Prometheus service.

### Cloudflare Tunnel Config (`~/.cloudflared/config.yml`)
```yaml
tunnel: grafana-local
credentials-file: ~/.cloudflared/grafana-local.json

ingress:
  - hostname: bay-bridge-traffic.com
    service: http://localhost:8080
  - service: http_status:404
```

## Troubleshooting

### Services Not Starting
```bash
# Check Docker
docker ps

# Check Nginx
ps aux | grep nginx

# Check ports
lsof -i :3000 -i :8080 -i :9090 -i :9091
```

### Tunnel Issues
```bash
# List tunnels
cloudflared tunnel list

# Check tunnel status
cloudflared tunnel info grafana-local

# View tunnel logs with debug output
cloudflared tunnel --loglevel debug run grafana-local
```

### Dashboard Not Loading
1. Check if Grafana is accessible: http://localhost:3000
2. Check if reverse proxy works: http://localhost:8080
3. Verify tunnel is running and DNS is configured

### Common Issues and Solutions

#### Issue 1: Tunnel Credentials File Not Found
**Error**: `Tunnel credentials file '/Users/username/.cloudflared/grafana-local.json' doesn't exist`

**Cause**: The setup script creates credentials with tunnel ID as filename, but config expects `grafana-local.json`

**Solution**: Update config file to use correct credentials path:
```bash
# Find the actual credentials file
ls ~/.cloudflared/*.json

# Update config.yml to use the correct file
sed -i '' 's|credentials-file: ~/.cloudflared/grafana-local.json|credentials-file: ~/.cloudflared/[TUNNEL_ID].json|' ~/.cloudflared/config.yml
```

#### Issue 2: DNS Routing Fails During Setup
**Error**: `Failed to add route: code: 1003, reason: Failed to create record... An A, AAAA, or CNAME record with that host already exists`

**Cause**: Existing DNS record conflicts with CNAME creation

**Solution**: Manually configure DNS in Cloudflare Dashboard:
1. Go to Cloudflare Dashboard → DNS → Records
2. Delete existing A/AAAA record for domain
3. Create new CNAME record:
   - **Name**: `bay-bridge-traffic.com` (or `@` for root)
   - **Target**: `[TUNNEL_ID].cfargotunnel.com`
   - **Proxy**: Enabled (orange cloud)

#### Issue 3: Grafana Authentication Issues
**Error**: `unexpected response with status code 401: authentication error`

**Cause**: Using Grafana Cloud write-only API key for read operations

**Solution**: Switch to local Prometheus data source:
```yaml
# Update grafana/provisioning/datasources/prometheus.yml
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://host.docker.internal:9090  # Local Prometheus
    uid: prometheus
    isDefault: true
    editable: false
```

#### Issue 4: Grafana Static Assets Not Loading
**Error**: "Grafana has failed to load its application files... caused by reverse proxy settings"

**Cause**: Reverse proxy configuration interfering with static asset loading

**Current Status**: ✅ **RESOLVED**

**Resolution**:
1. **Fixed Grafana Environment Variables**: Ensured `GF_SERVER_ROOT_URL` includes trailing slash
2. **Updated Nginx Configuration**:
   - Changed `server_name` to `_` (accept any hostname)
   - Set `proxy_set_header Host $http_host` to preserve original host
   - Added proper HTTPS headers for Cloudflare tunnel
3. **Container Recreation**: Full container recreation to apply environment variables

**Verified Working**:
- ✅ Static assets (CSS/JS) loading correctly
- ✅ Grafana logo and UI elements appear
- ✅ Cloudflare tunnel connectivity established

#### Issue 5: WebSocket and API Endpoint Failures
**Error**: WebSocket connections and some API endpoints failing through reverse proxy

**Current Status**: ⚠️ **ACTIVE ISSUE**

**Symptoms**:
- WebSocket connection to `ws://localhost:8080/api/live/ws` fails repeatedly
- API endpoint `/api/frontend-metrics` returns 404
- Grafana UI loads but shows continuous loading/connection issues
- Browser console shows repeated WebSocket connection failures

**Root Cause**:
- WebSocket upgrade headers may not be properly handled through Cloudflare tunnel
- Some API endpoints inconsistently accessible through reverse proxy
- Grafana Live features require persistent WebSocket connections

**Debug Information**:
```javascript
// Browser console errors:
WebSocket connection to 'ws://localhost:8080/api/live/ws' failed
POST http://localhost:8080/api/frontend-metrics 404 (Not Found)
```

**Attempted Solutions**:
- ✅ Added WebSocket upgrade headers to Nginx configuration
- ✅ Verified API endpoints work directly on Grafana (localhost:3000)
- ❌ WebSocket connections still fail through tunnel

**Next Steps**:
- Investigate Cloudflare tunnel WebSocket support limitations
- Consider disabling Grafana Live features for public access
- Test alternative WebSocket proxy configurations
- Evaluate if WebSocket failures affect core dashboard functionality

## Security Considerations

### ⚠️ **CRITICAL SECURITY NOTES**
- **API Keys**: The `grafana/provisioning/datasources/prometheus.yml` file contains sensitive Grafana Cloud API credentials
- **File Protection**: Ensure this file is not committed to public repositories or shared
- **Access Control**: Limit file system access to the grafana provisioning directory
- **Credential Rotation**: Regularly rotate Grafana Cloud API keys

### Cloudflare Tunnel Benefits
- **No Port Forwarding**: No need to open firewall ports
- **DDoS Protection**: Cloudflare's edge network protection
- **SSL/TLS Termination**: Automatic HTTPS with valid certificates
- **Access Control**: Can be configured with Cloudflare Access policies

### Authentication
- **Grafana**: Admin credentials (admin/admin) - **MUST be changed in production**
- **Cloudflare**: Uses OAuth flow for tunnel authentication
- **Prometheus**: Uses API key authentication to Grafana Cloud (stored in provisioning files)

### Network Security
- **Local Only**: All services bind to localhost only
- **Tunnel Encryption**: All traffic encrypted through Cloudflare Tunnel
- **No Direct Exposure**: No services directly exposed to internet

## Performance Considerations

### Latency
- **Additional Hops**: Request goes through Cloudflare → Nginx → Grafana
- **Local Processing**: Dashboard rendering happens locally for better performance
- **Data Source**: Remote Prometheus may add latency to queries

### Resource Usage
- **Grafana**: ~100-200MB RAM usage
- **Nginx**: Minimal resource usage (~10MB)
- **Cloudflared**: ~20-50MB RAM usage

### Scaling
- **Single User**: Current setup optimized for single user access
- **Concurrent Users**: May need resource adjustments for multiple users
- **Data Retention**: Grafana uses remote Prometheus, no local storage concerns

## Monitoring and Maintenance

### Health Checks
```bash
# Check all services
./start-bay-bridge-services.sh

# Individual service checks
curl -s http://localhost:3000/api/health    # Grafana health
curl -s http://localhost:8080              # Nginx proxy
curl -s http://localhost:9090/-/healthy    # Prometheus health
```

### Log Locations
- **Grafana**: Docker logs via `docker logs grafana`
- **Nginx**: `/opt/homebrew/var/log/nginx/`
- **Cloudflared**: Console output when running
- **Prometheus**: Docker logs via `docker logs prometheus`

### Backup Considerations
- **Grafana Config**: All configuration is in version control
- **Dashboard**: Stored in `grafana/dashboards/` directory
- **Tunnel Config**: Stored in `~/.cloudflared/` directory
- **Data**: All metrics data is in remote Prometheus (no local backup needed)

## Stopping Services

```bash
# Stop tunnel
# Ctrl+C in tunnel terminal

# Stop Nginx
nginx -s stop

# Stop Docker services
docker-compose down
```

## Future Enhancements

### Potential Improvements
1. **SSL Certificates**: Local SSL for development
2. **Authentication**: Integration with Cloudflare Access
3. **Multiple Dashboards**: Support for additional dashboards
4. **Alerting**: Grafana alerting configuration
5. **Backup Automation**: Automated configuration backups

### Monitoring Additions
1. **Service Health Dashboard**: Monitor the monitoring stack itself
2. **Performance Metrics**: Track proxy and tunnel performance
3. **Usage Analytics**: Dashboard access patterns and usage
