# RFD-008: Custom Landing Page Solution for Public Dashboard Access

## Overview

This RFD describes the implementation of a custom landing page solution to resolve public dashboard access issues while maintaining full functionality and adding custom branding/content.

## Problem Statement

The current reverse proxy setup (RFD-005) experiences persistent issues with Grafana static asset loading through the Cloudflare tunnel:

### Current Issues
- **Static Asset Loading Failures**: CSS/JS files return 404 through the tunnel despite working locally
- **Grafana Error Message**: "Grafana has failed to load its application files" 
- **Reverse Proxy Complexity**: Multiple layers (Cloudflare → Nginx → Grafana) causing header/routing conflicts
- **Limited Customization**: Cannot add custom content (disclaimers, about info) around the dashboard

### Root Cause Analysis
1. **Host Header Conflicts**: Grafana expects specific host headers for static asset routing
2. **Cloudflare Caching**: 404 responses being cached, preventing fixes from taking effect
3. **WebSocket Issues**: Grafana Live features failing through reverse proxy layers
4. **Content Security Policy**: Potential CSP conflicts with proxied static assets

## Alternative Solutions Evaluated

### Option 1: Grafana Cloud Public Dashboard ❌
**Status**: Not viable
- Public dashboards cannot access private data sources
- All panels show "No data" when made public
- Limited to public data sources only for security reasons

### Option 2: Static Dashboard Export + Custom Site ⚠️
**Status**: Complex but viable
- Requires periodic data export/generation
- Loss of real-time interactivity
- Significant development overhead
- Good for read-only use cases

### Option 3: Custom Landing Page + iframe Embedding ✅
**Status**: Recommended solution
- Leverages working local Grafana instance
- Allows custom branding and content
- Maintains full dashboard functionality
- Minimal infrastructure changes required

### Option 4: Grafana Embedding API ❌
**Status**: Not available
- `allow_embedding` setting not available in Grafana Cloud
- Would require self-hosted Grafana (already have local instance)
- More complex than iframe approach

## Recommended Solution: Option 3 - Custom Landing Page

### Architecture Overview

```
Internet → Cloudflare Tunnel → Nginx → Custom Landing Page
                                    ↓
                               iframe → Local Grafana (localhost:3000)
```

### Components

1. **Custom HTML Landing Page**
   - Hosted by Nginx at document root
   - Contains custom branding, disclaimers, about information
   - Embeds Grafana dashboard in responsive iframe

2. **Modified Nginx Configuration**
   - Serves custom HTML page at root (`/`)
   - Proxies Grafana requests to `/grafana/*` path
   - Handles static assets correctly for iframe content

3. **Local Grafana Instance** (unchanged)
   - Continues running on localhost:3000
   - Serves dashboard content to iframe
   - No reverse proxy complications for static assets

## Implementation Plan

### Phase 1: Create Custom Landing Page

#### 1.1 HTML Structure
```html
<!DOCTYPE html>
<html>
<head>
    <title>Bay Bridge Traffic Detection System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Custom styling for branding */
        body { margin: 0; font-family: Arial, sans-serif; }
        .header { background: #1f1f1f; color: white; padding: 20px; }
        .dashboard-container { height: calc(100vh - 200px); }
        .footer { background: #f5f5f5; padding: 20px; }
        iframe { width: 100%; height: 100%; border: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Bay Bridge Traffic Detection System</h1>
        <p>Real-time traffic monitoring and analytics</p>
    </div>
    
    <div class="dashboard-container">
        <iframe src="/grafana/d/bay-bridge-traffic/bay-bridge-traffic-detection-system?orgId=1&refresh=30s&kiosk=tv"></iframe>
    </div>
    
    <div class="footer">
        <p><strong>Disclaimer:</strong> This is a personal project for educational purposes.</p>
        <p><strong>About:</strong> Created by [Your Name] using motion detection and Prometheus monitoring.</p>
        <p><strong>Technology:</strong> OpenCV, Python, Prometheus, Grafana, Cloudflare Tunnel</p>
    </div>
</body>
</html>
```

#### 1.2 Responsive Design Features
- Mobile-friendly viewport
- Responsive iframe sizing
- Clean, professional appearance
- Customizable branding sections

### Phase 2: Update Nginx Configuration

#### 2.1 New Nginx Structure
```nginx
server {
    listen 8080;
    server_name _;

    # Serve custom landing page at root
    location = / {
        root /path/to/custom/pages;
        index index.html;
    }

    # Serve static assets for custom page
    location /static/ {
        root /path/to/custom/pages;
        expires 1d;
    }

    # Proxy Grafana under /grafana path
    location /grafana/ {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host localhost:3000;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        
        # Remove /grafana prefix when forwarding
        rewrite ^/grafana/(.*) /$1 break;
    }
}
```

#### 2.2 Configuration Benefits
- Clean separation between custom content and Grafana
- No host header conflicts for Grafana static assets
- Ability to serve custom static assets (CSS, images, etc.)
- Maintains existing tunnel configuration

### Phase 3: Dashboard URL Configuration

#### 3.1 Grafana Dashboard URL Parameters
- **Base URL**: `/grafana/d/[dashboard-uid]/[dashboard-slug]`
- **Kiosk Mode**: `?kiosk=tv` (removes Grafana UI chrome)
- **Auto-refresh**: `&refresh=30s` (automatic data refresh)
- **Organization**: `&orgId=1` (default organization)
- **Time Range**: `&from=now-1h&to=now` (optional)

#### 3.2 iframe Optimization
```html
<iframe 
    src="/grafana/d/bay-bridge-traffic/bay-bridge-traffic-detection-system?orgId=1&refresh=30s&kiosk=tv"
    frameborder="0"
    scrolling="no"
    allowfullscreen>
</iframe>
```

### Phase 4: Custom Content Integration

#### 4.1 Header Section
- Project title and description
- Real-time status indicators
- Navigation links (if needed)

#### 4.2 Footer Section
- Disclaimer about personal/educational use
- Technology stack information
- Contact/about information
- Links to documentation or source code

#### 4.3 Sidebar Options (Optional)
- System status indicators
- Quick stats summary
- External links

## File Structure

```
20250802_bay_bridge_traffic_cam/
├── nginx/
│   └── bay-bridge-traffic.conf          # Updated Nginx config
├── public/                               # New: Custom web content
│   ├── index.html                       # Main landing page
│   ├── static/
│   │   ├── style.css                    # Custom styling
│   │   ├── script.js                    # Optional JavaScript
│   │   └── images/                      # Logos, icons, etc.
│   └── favicon.ico                      # Custom favicon
└── docs/
    └── RFD-008-CUSTOM_LANDING_PAGE_SOLUTION.md
```

## Implementation Steps

### Step 1: Create Directory Structure
```bash
mkdir -p public/static/images
```

### Step 2: Create Landing Page
- Create `public/index.html` with custom content
- Add CSS styling in `public/static/style.css`
- Add any custom images/logos

### Step 3: Update Nginx Configuration
- Modify `nginx/bay-bridge-traffic.conf`
- Test configuration: `nginx -t`
- Reload: `nginx -s reload`

### Step 4: Test Local Access
- Verify custom page loads: `http://localhost:8080`
- Verify iframe content loads: Check dashboard within iframe
- Test responsive design on different screen sizes

### Step 5: Test Public Access
- Access via tunnel: `https://bay-bridge-traffic.com`
- Verify all functionality works through Cloudflare
- Test on multiple devices/browsers

## Benefits of This Approach

### Technical Benefits
- ✅ **Eliminates Static Asset Issues**: Grafana serves assets directly to iframe
- ✅ **Maintains Full Functionality**: All Grafana features work normally
- ✅ **Simple Architecture**: Fewer proxy layers and complications
- ✅ **Easy Debugging**: Clear separation between custom content and dashboard

### User Experience Benefits
- ✅ **Custom Branding**: Full control over page appearance
- ✅ **Additional Content**: Space for disclaimers, about info, etc.
- ✅ **Professional Appearance**: Clean, branded landing page
- ✅ **Mobile Responsive**: Works well on all device sizes

### Maintenance Benefits
- ✅ **Minimal Infrastructure Changes**: Reuses existing components
- ✅ **Easy Updates**: Custom content can be updated independently
- ✅ **Clear Separation**: Dashboard and custom content are separate
- ✅ **Fallback Options**: Can easily switch back to direct Grafana access

## Security Considerations

### iframe Security
- **X-Frame-Options**: Ensure Grafana allows iframe embedding
- **Content Security Policy**: Configure CSP headers appropriately
- **Same-Origin Policy**: Both custom page and Grafana served from same domain

### Access Control
- **Anonymous Access**: Grafana configured for anonymous viewing
- **Network Security**: All traffic still goes through Cloudflare tunnel
- **No Additional Exposure**: Same security model as current setup

## Testing Plan

### Local Testing
1. **Custom Page Loading**: Verify HTML/CSS loads correctly
2. **iframe Functionality**: Confirm dashboard displays in iframe
3. **Responsive Design**: Test on different screen sizes
4. **Static Assets**: Verify custom CSS/JS/images load

### Public Testing
1. **Tunnel Access**: Test via `https://bay-bridge-traffic.com`
2. **Cross-Browser**: Test on Chrome, Firefox, Safari, Edge
3. **Mobile Devices**: Test on phones and tablets
4. **Performance**: Monitor loading times and responsiveness

### Functionality Testing
1. **Dashboard Interactivity**: Verify all Grafana features work in iframe
2. **Auto-refresh**: Confirm data updates automatically
3. **Time Range Selection**: Test if time controls work (if not in kiosk mode)
4. **Panel Interactions**: Verify zoom, hover, click functionality

## Rollback Plan

If issues arise, easy rollback options:

### Option A: Revert to Direct Grafana
- Update Nginx to proxy all requests to Grafana
- Remove custom landing page
- Return to previous configuration

### Option B: Serve Dashboard Directly
- Configure Nginx root redirect to Grafana dashboard URL
- Bypass custom landing page temporarily
- Maintain tunnel functionality

## Future Enhancements

### Phase 2 Features
- **Real-time Status API**: Add endpoint for system health status
- **Multiple Dashboards**: Support for additional dashboard views
- **User Preferences**: Cookie-based settings for refresh rate, theme, etc.
- **Analytics**: Track usage patterns and popular features

### Advanced Features
- **Progressive Web App**: Add PWA manifest for mobile app-like experience
- **Offline Support**: Cache dashboard data for offline viewing
- **Custom Alerts**: Browser notifications for traffic anomalies
- **Social Sharing**: Share specific time ranges or interesting events

## Success Criteria

### Primary Goals
- ✅ Public dashboard accessible at `https://bay-bridge-traffic.com`
- ✅ All Grafana functionality works correctly
- ✅ Custom branding and content displayed
- ✅ No "failed to load application files" errors

### Secondary Goals
- ✅ Mobile-responsive design
- ✅ Professional appearance
- ✅ Fast loading times (<3 seconds)
- ✅ Cross-browser compatibility

This solution provides a robust, maintainable approach to public dashboard sharing while adding the desired custom content and branding capabilities.
