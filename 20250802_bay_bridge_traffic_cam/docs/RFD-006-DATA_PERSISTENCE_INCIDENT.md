# RFD-006: Data Persistence Incident and Resolution

## Overview

This RFD documents a critical data loss incident that occurred during reverse proxy debugging and the subsequent fixes implemented to prevent future data loss. The incident resulted in the loss of 6+ hours of traffic monitoring data due to improper Docker volume configuration.

## Incident Summary

**Date**: August 3, 2025  
**Duration**: Multiple data loss events over ~30 minutes  
**Impact**: Complete loss of historical Prometheus metrics data  
**Root Cause**: Missing persistent volume configuration for Prometheus container  

### Timeline of Events

1. **Initial Issue**: User reported Grafana loading errors ("Grafana has failed to load its application files")
2. **Debug Attempt**: Modified Grafana configuration and restarted containers
3. **First Data Loss**: 6+ hours of traffic data lost when `docker-compose down` destroyed Prometheus container
4. **Configuration Fix**: Added persistent volume configuration
5. **Second Data Loss**: Additional 15 minutes of data lost during second restart
6. **Resolution**: Persistent volumes now properly configured

## Technical Root Cause Analysis

### Original Configuration Problem

The original `docker-compose.yml` configuration lacked persistent storage for Prometheus:

```yaml
# PROBLEMATIC CONFIGURATION
prometheus:
  image: prom/prometheus:latest
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
  # MISSING: No persistent volume for /prometheus data directory
```

**What happened:**
- Prometheus stored TSDB data in `/prometheus` directory inside container
- No volume mount meant data existed only in container's ephemeral filesystem
- `docker-compose down` destroyed container and all data permanently

### Data Loss Mechanism

1. **Container Storage**: Prometheus data stored in container's `/prometheus` directory
2. **No Persistence**: No Docker volume mounted to preserve data
3. **Container Destruction**: `docker-compose down` removes container and all internal data
4. **Data Recovery**: Impossible - data only existed in destroyed container

### Why Persistent Volumes Didn't Help Initially

The sequence of events explains why adding persistent volume configuration didn't prevent the second data loss:

1. **First restart**: No volume config → data lost
2. **Added volume config**: Configuration updated but volume doesn't exist yet
3. **Second restart**: `docker-compose down` destroys container before volume is created
4. **Volume creation**: Only happens during `docker-compose up` with new config

## Resolution and Fixes

### 1. Persistent Volume Configuration

Updated `docker-compose.yml` to include persistent storage:

```yaml
# FIXED CONFIGURATION
prometheus:
  image: prom/prometheus:latest
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    - prometheus-storage:/prometheus  # Persistent volume added
  command:
    - '--storage.tsdb.path=/prometheus'
    - '--storage.tsdb.retention.time=200h'

volumes:
  prometheus-storage:  # Named volume definition
  grafana-storage:
```

### 2. Volume Storage Details

- **Volume Name**: `20250802_bay_bridge_traffic_cam_prometheus-storage`
- **Mount Point**: `/prometheus` (inside container)
- **Host Location**: `/var/lib/docker/volumes/20250802_bay_bridge_traffic_cam_prometheus-storage/_data`
- **Retention**: 200 hours (configured in Prometheus)

### 3. Grafana Configuration Fixes

During debugging, several Grafana configuration issues were also resolved:

#### Datasource Configuration
Fixed typo in `grafana/provisioning/datasources/prometheus.yml`:
```yaml
# BEFORE (broken)
isDefault: truee

# AFTER (fixed)
isDefault: true
```

#### Dashboard Datasource References
Updated all dashboard panels to use local Prometheus instead of Grafana Cloud:
```json
// BEFORE (broken)
"datasource": {
  "uid": "grafanacloud-prom"
}

// AFTER (fixed)
"datasource": {
  "uid": "prometheus"
}
```

## Prevention Measures

### 1. Operational Procedures

**NEVER run `docker-compose down` on production systems without:**
1. Verifying all data is properly persisted
2. Creating backups of critical data
3. Understanding the impact of container destruction
4. Getting explicit approval for potentially destructive operations

### 2. Configuration Standards

**All stateful services MUST have persistent volumes:**
```yaml
# Template for stateful services
service-name:
  volumes:
    - service-data:/data/path
    
volumes:
  service-data:
```

### 3. Monitoring and Alerts

**Recommended additions:**
- Volume usage monitoring
- Data retention alerts
- Container restart notifications
- Backup verification checks

## Lessons Learned

### 1. Docker Volume Behavior

- **Named volumes persist** across container restarts
- **Anonymous volumes are destroyed** with containers
- **Volume creation timing** matters - volumes must exist before they can preserve data
- **`docker-compose down`** is destructive for non-persistent data

### 2. Configuration Management

- **Persistent storage** must be configured from the beginning
- **Retroactive persistence** doesn't recover already-lost data
- **Configuration changes** require careful sequencing to avoid data loss

### 3. Operational Practices

- **Always assume data loss** when restarting containers without persistent volumes
- **Test persistence** in development before production deployment
- **Document data storage** requirements for all services
- **Implement backup strategies** for critical data

## Current Status

✅ **Persistent volumes configured** for both Prometheus and Grafana  
✅ **Grafana datasource issues resolved**  
✅ **Dashboard connectivity restored**  
❌ **Historical data lost** (6+ hours of traffic metrics)  
✅ **Future data will persist** across container restarts  

## Recommendations

### Immediate Actions
1. **Monitor new data collection** to verify persistence works
2. **Test container restart** to confirm data survives
3. **Document backup procedures** for volume data

### Long-term Improvements
1. **Implement automated backups** of Docker volumes
2. **Add monitoring** for data retention and volume usage
3. **Create runbooks** for safe container management
4. **Consider external storage** for critical metrics (e.g., remote write to cloud)

## Technical Reference

### Volume Management Commands
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect 20250802_bay_bridge_traffic_cam_prometheus-storage

# Backup volume
docker run --rm -v prometheus-storage:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz -C /data .

# Restore volume
docker run --rm -v prometheus-storage:/data -v $(pwd):/backup alpine tar xzf /backup/prometheus-backup.tar.gz -C /data
```

### Data Recovery Commands

#### Failed API Token Approaches
```bash
# Direct Prometheus API (failed - scope issues)
curl -u "USER_ID:$GRAFANA_READ_TOKEN" \
  "https://prometheus-prod-XX-prod-us-west-0.grafana.net/api/prom/api/v1/label/__name__/values"

# Grafana API with Bearer token (failed - invalid key)
curl -H "Authorization: Bearer $GRAFANA_READ_TOKEN" \
  "https://USERNAME.grafana.net/api/datasources"
```

#### Working Session-based Approach
```bash
# Extract session cookie from browser Developer Tools
# Use captured grafana_session cookie for authenticated requests
curl -X POST "https://USERNAME.grafana.net/api/ds/query?ds_type=prometheus" \
  -H "content-type: application/json" \
  -H "cookie: grafana_session=<session_id>; grafana_session_expiry=<expiry>" \
  -H "x-grafana-org-id: 1" \
  -H "x-datasource-uid: grafanacloud-prom" \
  -d '{
    "queries": [
      {
        "refId": "A",
        "expr": "traffic_vehicles_total",
        "range": true,
        "format": "time_series",
        "start": <start_timestamp>,
        "end": <end_timestamp>,
        "step": 60
      }
    ]
  }'
```

### Safe Restart Procedure
```bash
# Safe restart with persistent volumes
docker-compose restart prometheus grafana

# Only use 'down' if you understand the implications
docker-compose down  # DANGEROUS without persistent volumes
docker-compose up -d
```

## Data Recovery Investigation

### Attempted Recovery Methods

Following the data loss incident, several methods were investigated to recover the lost historical data from Grafana Cloud:

#### 1. Direct Prometheus API Access
**Attempted**: Direct queries to Grafana Cloud Prometheus endpoint
- **Endpoint**: `https://prometheus-prod-XX-prod-us-west-0.grafana.net/api/prom/api/v1/label/__name__/values`
- **Authentication**: Basic auth with username `USER_ID` and various API tokens
- **Result**: Failed with "authentication error: invalid scope requested"
- **Tokens Tested**:
  - Write-only token: `glc_eyJ...` (expected to fail)
  - Read token: `glc_eyJ...` (created with read permissions)

#### 2. Grafana HTTP API Access
**Attempted**: Access via Grafana Cloud API endpoints
- **Endpoint**: `https://USERNAME.grafana.net/api/datasources`
- **Authentication**: Bearer token with read permissions
- **Result**: Failed with "Invalid API key" despite valid access policy
- **Access Policy**: Confirmed active with scopes: `metrics:read`, `datasources:read`, `alerts:read`, etc.

#### 3. Browser Session Authentication Discovery
**Breakthrough**: Captured working authentication from browser Developer Tools
- **Method**: Intercepted successful API calls from Grafana dashboard
- **Authentication**: Session-based using `grafana_session` cookie
- **Endpoint**: `POST /api/ds/query?ds_type=prometheus`
- **Status**: Promising approach for data extraction

### Key Findings

1. **API Token Limitations**: Standard API tokens appear insufficient for Prometheus data access
2. **Session Authentication**: Browser sessions use different auth mechanism than API tokens
3. **Data Accessibility**: Historical data exists and is queryable through Grafana interface
4. **Recovery Feasibility**: Data recovery is technically possible using session-based authentication

### Next Steps for Data Recovery

1. **Session-based Extraction**: Use captured session cookies to query historical data
2. **Time Range Targeting**: Focus on lost data period (6+ hours from incident)
3. **Data Format Conversion**: Convert extracted JSON to Prometheus TSDB format
4. **Local Import**: Import recovered data into local Prometheus instance

## Conclusion

This incident highlights the critical importance of proper data persistence configuration in containerized environments. While the immediate issue has been resolved, the investigation revealed that data recovery from cloud sources is technically feasible but requires session-based authentication rather than API tokens.

The implemented fixes ensure that future container restarts will preserve data, and the recovery investigation provides a pathway for retrieving lost historical data when needed.
