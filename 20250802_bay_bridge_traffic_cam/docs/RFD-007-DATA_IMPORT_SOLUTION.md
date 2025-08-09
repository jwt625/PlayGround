# RFD-007: Data Import Solution for Local Prometheus

## Overview

This RFD addresses the need to import backed up Prometheus data from Grafana Cloud into the local Prometheus container. Following the data persistence incident documented in RFD-006, we now have export tools but lack the capability to restore or import historical data into our local Prometheus instance.

## Problem Statement

**Current State:**
- ✅ Export tools available (`export_prometheus_data.py`, `export_dashboard_metrics.py`)
- ✅ Local Prometheus container with persistent storage
- ❌ No mechanism to import backed up data into local Prometheus

**Use Cases:**
1. **Data Recovery**: Restore lost data from backups after incidents
2. **Migration**: Move historical data from cloud to local instance
3. **Analysis**: Import specific time ranges for detailed analysis
4. **Testing**: Load test data into development environments

## Technical Challenges

### Prometheus Data Import Limitations

Prometheus TSDB is designed for real-time ingestion, not historical data import:

1. **No Native Import API**: Prometheus lacks a direct data import endpoint
2. **Time-based Ingestion**: Metrics are expected to arrive in near real-time
3. **TSDB Structure**: Direct file manipulation is complex and risky
4. **Timestamp Constraints**: Historical timestamps may be rejected

### Available Approaches

#### 1. Remote Write Replay ⭐ **Recommended**
- **Method**: Replay backed up data via remote write protocol
- **Pros**: Uses standard Prometheus ingestion path, safe, preserves metadata
- **Cons**: Requires custom tooling, may have rate limits

#### 2. Pushgateway Import
- **Method**: Send data through Prometheus Pushgateway
- **Pros**: Simple HTTP interface, good for batch data
- **Cons**: Limited to latest values, loses historical timestamps

#### 3. Direct TSDB Manipulation
- **Method**: Stop Prometheus and manipulate TSDB files directly
- **Pros**: Can preserve exact timestamps
- **Cons**: Risky, complex, requires Prometheus downtime

#### 4. Recording Rules Backfill
- **Method**: Use Prometheus recording rules with backfill capability
- **Pros**: Native Prometheus feature
- **Cons**: Limited to rule-based calculations, not raw data import

## Solution Design

### Primary Solution: Remote Write Replay Tool

Create `import_prometheus_data.py` with multiple import strategies:

```python
# Usage examples
python import_prometheus_data.py --input traffic_data.txt --method simulation
python import_prometheus_data.py --input-dir backup_20250803/ --method pushgateway
python import_prometheus_data.py --input traffic_data.txt --method http-server
python import_prometheus_data.py --input traffic_data.txt --method direct-write
python import_prometheus_data.py --input traffic_data.txt --method remote-write  # Future
```

### Import Methods

#### 1. Simulation Mode (Default)
- **Purpose**: Validate data format and show import preview
- **Safety**: No actual data modification
- **Output**: Data validation report and import statistics

#### 2. Pushgateway Mode
- **Purpose**: Import via Prometheus Pushgateway
- **Requirements**: Pushgateway service running
- **Limitations**: Only latest values, no historical timestamps

#### 3. HTTP Server Mode
- **Purpose**: Create static files for historical data import
- **Requirements**: HTTP server to serve metric files
- **Benefits**: Preserves original timestamps, can be scraped by Prometheus

#### 4. Direct Write Mode
- **Purpose**: Create TSDB blocks with original timestamps
- **Requirements**: promtool and Prometheus downtime
- **Benefits**: Direct TSDB manipulation with timestamp preservation
- **Status**: ❌ **FAILED** - Blocks created but not loaded by Prometheus

#### 5. Remote Write Mode (Future)
- **Purpose**: Direct remote write to Prometheus
- **Requirements**: Prometheus remote write endpoint enabled
- **Benefits**: Preserves timestamps and full metadata

### Data Flow Architecture

```
Backed Up Data (.txt files)
         ↓
   Import Tool Parser
         ↓
    Method Selection
    ↓         ↓         ↓         ↓         ↓
Simulation  Pushgateway  HTTP-Server  Direct-Write  Remote Write
    ↓         ↓         ↓         ↓         ↓
 Validation  HTTP POST   Static Files  TSDB Blocks  Protobuf/Snappy
    ↓         ↓         ↓         ↓         ↓
  Report    Pushgateway  HTTP Server   promtool    Prometheus
              ↓         ↓         ↓         ↓
           Prometheus   Prometheus   ❌ FAILED    TSDB
```

## Implementation Plan

### Phase 1: Basic Import Tool ✅ **COMPLETED**
- [x] Create `import_prometheus_data.py` script
- [x] Implement Prometheus format parser
- [x] Add simulation mode for validation
- [x] Add pushgateway mode for basic import
- [x] Command-line interface with multiple options

### Phase 2: Enhanced Import Capabilities ⚠️ **IN PROGRESS**
- [x] Implement HTTP server method for historical data files
- [x] Create TSDB blocks using promtool
- [x] Add direct-write method for timestamp preservation
- [ ] ❌ **FAILED**: TSDB block import method (blocks not loaded by Prometheus)
- [ ] Implement proper remote write protocol support
- [ ] Add protobuf and snappy compression
- [ ] Batch processing optimization
- [ ] Progress reporting and resume capability
- [ ] Data deduplication and conflict resolution

### Phase 3: Integration and Automation
- [ ] Docker Compose integration for Pushgateway
- [ ] Automated backup/restore workflows
- [ ] Grafana dashboard for import monitoring
- [ ] Documentation and runbooks

## Usage Guide

### Prerequisites

1. **Local Prometheus Running**:
   ```bash
   docker-compose up -d prometheus
   ```

2. **Pushgateway (for pushgateway method)**:
   ```bash
   docker run -d -p 9091:9091 prom/pushgateway
   ```

3. **Backed Up Data**:
   - Export data using existing tools
   - Ensure `.txt` format (Prometheus exposition format)

### Basic Usage

#### Validate Data Format
```bash
python scripts/import_prometheus_data.py --input traffic_vehicles_total.txt
```

#### Import Single File via Pushgateway
```bash
python scripts/import_prometheus_data.py \
  --input traffic_vehicles_total.txt \
  --method pushgateway \
  --prometheus-url http://localhost:9090
```

#### Import Directory of Files
```bash
python scripts/import_prometheus_data.py \
  --input-dir dashboard_export_20250803_120000/ \
  --method pushgateway
```

### Advanced Configuration

#### Environment Variables
```bash
# .env file
LOCAL_PROMETHEUS_URL=http://localhost:9090
PUSHGATEWAY_URL=http://localhost:9091
```

#### Custom Pushgateway URL
```bash
python scripts/import_prometheus_data.py \
  --input traffic_data.txt \
  --method pushgateway \
  --pushgateway-url http://custom-pushgateway:9091
```

## Data Format Requirements

### Input Format (Prometheus Exposition)
```
# HELP traffic_vehicles_total Total vehicles counted
# TYPE traffic_vehicles_total counter
traffic_vehicles_total{direction="left"} 1234 1754207395000
traffic_vehicles_total{direction="right"} 5678 1754207395000
```

### Supported Metrics
- Counter metrics (e.g., `traffic_vehicles_total`)
- Gauge metrics (e.g., `system_status`)
- Histogram metrics (e.g., `frame_processing_time_seconds`)

## Recent Progress and Findings

### August 3, 2025 - Historical Data Restoration Attempt

#### **Attempted Methods**

1. **✅ HTTP Server Method** - Successfully implemented
   - Created 1,210 individual metric files with original timestamps
   - Generated HTTP server to serve historical data
   - Files properly formatted in Prometheus exposition format
   - **Status**: Working but requires manual Prometheus configuration

2. **❌ Direct TSDB Block Import** - Failed
   - Used `promtool tsdb create-blocks-from openmetrics` to create 1,210 TSDB blocks
   - Successfully copied blocks to Prometheus data directory (Docker volume)
   - Fixed ownership and permissions (UID 65534)
   - **Issue**: Prometheus does not load/recognize the imported blocks
   - **Result**: No historical data queryable despite blocks being present

#### **Technical Details of TSDB Block Failure**

**What Worked:**
- ✅ `promtool` successfully created TSDB blocks from OpenMetrics data
- ✅ Blocks copied to correct Prometheus data directory (`prometheus-storage` volume)
- ✅ Proper file ownership and permissions set
- ✅ Block metadata shows correct timestamps and metrics

**What Failed:**
- ❌ Prometheus does not discover or load the imported blocks
- ❌ Historical data not queryable via Prometheus API
- ❌ No errors in Prometheus logs about block loading
- ❌ Grafana dashboard shows no historical data

**Root Cause Analysis:**
- TSDB block import requires Prometheus to be stopped during import
- Prometheus may have strict requirements for block consistency and ordering
- Imported blocks may conflict with existing TSDB structure
- Time range of imported data (August 2025) may be outside expected ranges

#### **Current Status**

**Data Available:**
- ✅ Current live traffic data (post-data-loss)
- ✅ Exported historical data in multiple formats
- ✅ HTTP server with historical data files ready for import

**Data Missing:**
- ❌ Historical data in Prometheus TSDB (pre-data-loss)
- ❌ Historical data visible in Grafana dashboards
- ❌ Time series continuity across the data loss incident

#### **Lessons Learned**

1. **TSDB Block Import is Unreliable**: Direct manipulation of Prometheus TSDB blocks is complex and error-prone
2. **Pushgateway Limitations**: Only stores current values, not historical time series
3. **Timestamp Preservation**: Most import methods lose original timestamps
4. **Docker Volume Complexity**: Data persistence in containerized environments requires careful volume management

#### **Recommended Next Steps**

1. **Accept Data Loss**: Acknowledge that historical data restoration is not feasible with current tools
2. **Improve Backup Strategy**: Implement more robust backup mechanisms (remote write, federation)
3. **Focus on Prevention**: Ensure data persistence is properly configured going forward
4. **Alternative Approaches**: Consider external time series databases for long-term storage

## Limitations and Considerations

## Data Overlap Handling

### What Happens with Overlapping Data?

When import data overlaps with existing Prometheus data, different behaviors occur depending on the import method:

#### **Pushgateway Method**
- **Behavior**: Overwrites existing metrics with same job/instance labels
- **Result**: Latest imported value wins, no duplicates
- **Limitation**: Historical timestamps are lost (only current values stored)
- **Use Case**: Good for correcting recent data or adding missing metrics

#### **Remote Write Method** (future)
- **Behavior**: Prometheus TSDB keeps latest value for each exact timestamp
- **Result**: Exact timestamp matches get overwritten
- **Consideration**: Close timestamps may create multiple data points
- **Use Case**: Best for historical data restoration with timestamp preservation

#### **Direct TSDB Method**
- **Behavior**: Depends on implementation, potentially dangerous
- **Result**: Could corrupt data or fail completely
- **Recommendation**: Avoid unless absolutely necessary

### Overlap Detection and Strategies

The import tool now includes automatic overlap detection:

```bash
# Check for overlaps (default behavior)
python scripts/import_prometheus_data.py --input data.txt

# Skip overlap checking for faster imports
python scripts/import_prometheus_data.py --input data.txt --skip-overlap-check

# Control overlap behavior
python scripts/import_prometheus_data.py --input data.txt --overlap-strategy fail
```

#### **Overlap Strategies**

1. **`warn` (default)**: Detect and warn about overlaps, but proceed with import
2. **`skip`**: Skip importing data points that would overlap with existing data
3. **`overwrite`**: Proceed with import, explicitly overwriting existing data
4. **`fail`**: Stop import if any overlaps are detected

#### **Overlap Detection Process**

1. **Time Range Analysis**: Determine start/end times of import data
2. **Metric Query**: Check existing Prometheus data for same metrics in same time range
3. **Conflict Identification**: Identify specific metrics and time periods with overlaps
4. **Strategy Application**: Apply chosen overlap strategy
5. **Recommendation Generation**: Provide specific guidance based on findings

### Current Limitations
1. **Pushgateway Method**: Only preserves latest values, not full time series
2. **TSDB Block Import**: Blocks created but not loaded by Prometheus (method unreliable)
3. **Timestamp Handling**: Historical timestamps may not be preserved in all methods
4. **Rate Limiting**: Large imports may need throttling
5. **Memory Usage**: Large files require careful batch processing
6. **Overlap Detection**: Limited to first 3 metrics to avoid overwhelming Prometheus
7. **Docker Volume Complexity**: TSDB blocks must be in correct Docker volume with proper ownership

### Best Practices
1. **Start with Simulation**: Always validate data before importing
2. **Small Batches**: Import data in manageable chunks
3. **Monitor Resources**: Watch Prometheus memory and disk usage
4. **Backup First**: Backup current Prometheus data before importing
5. **Verify Results**: Check imported data in Grafana dashboards

## Future Enhancements

### Remote Write Protocol Support
- Implement proper Prometheus remote write protocol
- Add protobuf serialization and snappy compression
- Support for historical timestamp preservation

### Advanced Features
- **Incremental Import**: Only import new/changed data
- **Data Validation**: Verify data integrity and consistency
- **Conflict Resolution**: Handle duplicate or conflicting metrics
- **Progress Tracking**: Resume interrupted imports

### Integration Improvements
- **Grafana Integration**: Import status dashboards
- **Alerting**: Notifications for import success/failure
- **Automation**: Scheduled backup and restore workflows

## Testing Strategy

### Unit Tests
- Data parser validation
- Format conversion accuracy
- Error handling scenarios

### Integration Tests
- End-to-end import workflows
- Prometheus connectivity
- Pushgateway integration

### Performance Tests
- Large file import performance
- Memory usage optimization
- Concurrent import handling

## Security Considerations

### Access Control
- Prometheus and Pushgateway authentication
- Secure credential management
- Network access restrictions

### Data Validation
- Input sanitization
- Metric name validation
- Value range checking

## Monitoring and Observability

### Import Metrics
- Import success/failure rates
- Processing time and throughput
- Data volume imported
- Error categorization

### Alerting
- Failed import notifications
- Resource usage alerts
- Data quality warnings

## Conclusion

The data import solution provides a comprehensive approach to restoring backed up Prometheus data to local instances. The phased implementation starts with safe validation and basic import capabilities, with plans for enhanced remote write support and automation.

**Key Benefits:**
- ✅ Safe data validation before import
- ✅ Multiple import methods for different use cases
- ✅ Integration with existing export tools
- ✅ Extensible architecture for future enhancements

**Next Steps:**
1. ✅ ~~Test the basic import tool with existing backup data~~ **COMPLETED**
2. ✅ ~~Set up Pushgateway for production imports~~ **COMPLETED**
3. ❌ ~~Develop TSDB block import~~ **FAILED - Method unreliable**
4. [ ] Develop remote write protocol support
5. [ ] Create automated backup/restore workflows
6. [ ] Implement alternative backup strategies (federation, remote write)
7. [ ] Accept current data loss and focus on prevention

This solution addresses the critical gap identified in RFD-006 and provides a foundation for robust data management in the Bay Bridge Traffic monitoring system.
