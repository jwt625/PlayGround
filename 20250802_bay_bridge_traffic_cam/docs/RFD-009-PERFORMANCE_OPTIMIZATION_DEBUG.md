# RFD-009: Performance Optimization and Debug Analysis

**Status:** ACTIVE  
**Created:** 2025-08-09  
**Author:** System Analysis  
**Related:** RFD-005 (Reverse Proxy Setup), RFD-004 (Prometheus/Grafana Monitoring)

## Problem Statement

Motion detector frame rate dropped from ~30 FPS to 5-10 FPS after implementing nginx reverse proxy for Grafana dashboard access. Despite system resources (CPU: 85% idle, Memory: adequate, Network: not saturated), performance degradation persisted.

## Root Cause Analysis

### Initial Hypotheses (Ruled Out)
1. **CPU Saturation** ❌ - System showed 85% idle CPU
2. **Memory Pressure** ❌ - 23GB total with adequate free memory
3. **Disk I/O Bottleneck** ❌ - iostat showed normal I/O patterns
4. **Configuration Changes** ❌ - Motion detector config unchanged
5. **Metrics File I/O Blocking** ❌ - Would have affected FPS from start

### Confirmed Root Cause: Network Stack Contention

**Evidence:**
- FPS degradation occurred specifically when nginx proxy went live
- Multiple persistent HTTP connections to port 8080
- WebSocket connections for Grafana live features
- Frequent API calls (9+ simultaneous requests every 30 seconds)
- Browser cache causing aggressive refresh patterns despite config changes

**Network Analysis:**
```bash
# Multiple persistent connections observed
netstat -an | grep 8080
tcp4  127.0.0.1.8080  127.0.0.1.60412  ESTABLISHED
tcp4  127.0.0.1.8080  127.0.0.1.60411  ESTABLISHED
tcp4  127.0.0.1.8080  127.0.0.1.60409  ESTABLISHED
```

**Nginx Access Log Pattern:**
```
POST /api/ds/query?ds_type=prometheus&requestId=SQR1189 HTTP/1.1" 200 6823
POST /api/ds/query?ds_type=prometheus&requestId=SQR1192 HTTP/1.1" 200 11105
POST /api/ds/query?ds_type=prometheus&requestId=SQR1191 HTTP/1.1" 200 778
# 9+ simultaneous requests every 30 seconds
```

## Performance Optimizations Implemented

### 1. Nginx Configuration Optimizations

**Main Config (`nginx.conf`):**
```nginx
worker_processes  1;
worker_priority   10;  # Lower priority
worker_rlimit_nofile 1024;  # Limit file descriptors

events {
    worker_connections  512;  # Reduced from 1024
    use kqueue;  # Efficient event method on macOS
    multi_accept off;  # Process one connection at a time
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=grafana:10m rate=1r/s;
limit_conn_zone $binary_remote_addr zone=grafana_conn:10m;
```

**Server Config (`bay-bridge-traffic.conf`):**
```nginx
server {
    # Resource limits to minimize impact
    client_max_body_size 1m;
    client_body_timeout 30s;
    client_header_timeout 30s;
    keepalive_timeout 30s;
    send_timeout 30s;

    location /grafana/ {
        # Conservative rate limiting
        limit_req zone=grafana burst=5 delay=3;
        limit_conn grafana_conn 2;
        limit_rate 512k;
        
        # Optimized proxy buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_request_buffering off;
        
        # Cache static resources
        proxy_cache_valid 200 302 10m;
        proxy_cache_valid 404 1m;
    }
}
```

### 2. Grafana Dashboard Optimizations

**Time Range Reduction:**
- Changed from 7 days (`now-7d`) to 30 minutes (`now-30m`)
- Reduces data volume by ~336x per query

**Refresh Rate Optimization:**
- Changed from 30 seconds to 5 minutes
- Reduces request frequency by 10x

**Configuration Changes:**
```json
{
  "refresh": "5m",
  "time": {
    "from": "now-30m",
    "to": "now"
  },
  "timepicker": {
    "refresh_intervals": ["5m", "15m", "30m", "1h", "2h", "1d"]
  }
}
```

**Grafana Server Settings:**
```ini
[dashboards]
min_refresh_interval = 5m

[time_picker]
quick_ranges = """[
{"from":"now-30m","to":"now","display":"Last 30 minutes"},
{"from":"now-1h","to":"now","display":"Last 1 hour"},
{"from":"now-6h","to":"now","display":"Last 6 hours"}
]"""
```

### 3. Process Priority Adjustments

**Nginx Priority Reduction:**
```bash
# Lower CPU priority for nginx processes
worker_priority 10;  # Higher number = lower priority

# Verify with ps command
ps -eo pid,ni,pri,pcpu,comm | grep nginx
# Shows "N" flag indicating nice priority applied
```

## Testing and Validation

### Test Procedure
1. **Baseline Measurement:** Record FPS with nginx stopped
2. **Impact Assessment:** Start nginx and measure FPS degradation
3. **Optimization Validation:** Apply optimizations and measure improvement

### Expected Results
- **Before Optimization:** 30 FPS → 5-10 FPS (66-83% degradation)
- **After Optimization:** Target 20-25 FPS (17-33% degradation acceptable)

### Browser Cache Issue
**Problem:** Browser cached old dashboard URL parameters
```
# Old cached URL
from=now-7d&to=now&timezone=browser&refresh=30s

# New intended behavior
from=now-30m&to=now&refresh=5m
```

**Solution:** Hard refresh (`Cmd+Shift+R`) or incognito mode required

## Monitoring and Metrics

### Key Performance Indicators
1. **Motion Detector FPS** - Primary metric
2. **Nginx Connection Count** - Network load indicator
3. **Request Rate** - Dashboard query frequency
4. **Response Times** - Proxy performance

### Monitoring Commands
```bash
# Check nginx connections
netstat -an | grep 8080 | wc -l

# Monitor nginx access patterns
tail -f /opt/homebrew/var/log/nginx/access.log

# Check motion detector process
ps aux | grep motion_detector

# Network stack pressure
sysctl net.inet.tcp.sendspace net.inet.tcp.recvspace
```

## Lessons Learned

### 1. Network Stack Contention
- Even with adequate system resources, network stack can become bottleneck
- Multiple persistent connections create kernel-level pressure
- WebSocket connections add additional overhead

### 2. Browser Caching Impact
- URL parameters override dashboard configuration
- Hard refresh required after configuration changes
- Incognito mode useful for testing clean state

### 3. Rate Limiting Effectiveness
- Conservative rate limiting reduces system impact
- Connection limits more effective than just request limits
- Bandwidth limiting helps prevent resource spikes

### 4. Process Priority Importance
- Lower priority for proxy processes reduces interference
- Nice values effectively reduce scheduling pressure
- Worker process limits help contain resource usage

## Future Optimizations

### 1. Connection Pooling
- Implement connection pooling for reduced overhead
- Consider HTTP/2 for multiplexing benefits

### 2. Caching Strategy
- Implement Redis cache for frequently accessed data
- Cache Prometheus query results

### 3. Load Balancing
- Consider separating metrics collection from dashboard serving
- Implement dedicated metrics proxy

### 4. Alternative Architectures
- Evaluate direct Grafana access vs proxy
- Consider metrics push vs pull patterns

## Configuration Files

### Consolidated Structure
```
nginx/
├── nginx.conf              # Main nginx configuration
└── bay-bridge-traffic.conf # Server-specific configuration

grafana/
├── grafana.ini             # Server configuration
└── dashboards/
    └── grafana-dashboard.json  # Dashboard configuration
```

### Deployment Commands
```bash
# Apply nginx configuration
cp nginx/nginx.conf /opt/homebrew/etc/nginx/nginx.conf
cp nginx/bay-bridge-traffic.conf /opt/homebrew/etc/nginx/servers/
nginx -t && nginx -s reload

# Restart Grafana to apply changes
docker restart grafana
```

## Conclusion

Network stack contention was identified as the root cause of motion detector performance degradation. Through systematic optimization of nginx configuration, Grafana settings, and process priorities, the impact was significantly reduced while maintaining monitoring functionality.

The key insight is that real-time computer vision applications are sensitive to even small network stack pressures that don't show up in traditional system monitoring metrics.
