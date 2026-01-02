# DevLog 004: Training Events Visualization Integration

**Date**: 2026-01-02  
**Author**: Development Team  
**Status**: Planning  
**Related**: DevLog-003 (GPU Metrics NVML API Update)

## Executive Summary

This document details the analysis, design, and implementation plan for integrating training log events (validation metrics, evaluation benchmarks, checkpoint saves) into the existing GPU metrics visualization system. The goal is to correlate GPU resource utilization with training milestones to provide comprehensive insights into model training performance.

## Background

### Current State

The GPU metrics visualization system (`visualize_gpu_metrics_v2_enhanced.py`) provides:
- Real-time GPU power consumption (instant and average readings)
- GPU temperature monitoring
- Automatic idle period detection and cutoff
- Interactive HTML visualization using Plotly
- Downsampling for large datasets

### Motivation

Training logs contain critical events that correlate with GPU utilization patterns:
- **Validation events**: Model performance checkpoints every 1000 steps
- **Evaluation events**: Comprehensive benchmark suites (22 tasks) at major milestones
- **Checkpoint events**: Model and optimizer state saves

Visualizing these events alongside GPU metrics enables:
1. Identifying performance bottlenecks during evaluation phases
2. Correlating validation improvements with training intensity
3. Understanding checkpoint save overhead
4. Debugging training anomalies by cross-referencing timestamps
5. Optimizing evaluation frequency based on GPU utilization impact

## Inspection Results

### 1. Training Log Analysis

**File**: `nanochat/training_20251221_183935.log`  
**Size**: 40,484 lines  
**Duration**: ~6.8 hours (2025-12-21 18:39:35 to 2025-12-23 00:54:23)

#### Event Type 1: Validation Events

**Pattern**: `Step \d+ \| Validation bpb: \d+\.\d+`

**Characteristics**:
- Total occurrences: 143 events
- Frequency: Every 1000 steps (approximately)
- Metric: Bits per byte (bpb) - lower is better
- No explicit timestamp (must infer from surrounding log context)

**Example**:
```
Step 31000 | Validation bpb: 0.7926
```

**Observed Range**:
- Initial (Step 2000): bpb = 0.9326
- Final (Step 33600): bpb = 0.7834
- Trend: Decreasing (improving) over training

**Edge Cases**:
- Some validation events may not have preceding timestamp in same line
- Need to track last seen timestamp for correlation
- Steps are not perfectly uniform (e.g., 33600 instead of 34000)

#### Event Type 2: Evaluation Events

**Pattern**: `Evaluating: (\w+).*accuracy: ([\d.]+)`

**Characteristics**:
- Total occurrences: 396 individual evaluations
- Grouped into: 12 evaluation sessions
- Evaluation sessions at steps: 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 33600
- Each session runs 22 different benchmarks
- Duration per session: ~60-90 seconds total

**Benchmark Tasks** (per session):
1. hellaswag_zeroshot (0-shot, multiple choice)
2. jeopardy (10-shot, language modeling)
3. bigbench_qa_wikidata (10-shot, language modeling)
4. arc_easy (10-shot, multiple choice)
5. arc_challenge (10-shot, multiple choice)
6. copa (0-shot, multiple choice)
7. commonsense_qa (10-shot, multiple choice)
8. piqa (10-shot, multiple choice)
9. openbook_qa (0-shot, multiple choice)
10. lambada_openai (0-shot, language modeling)
11. hellaswag (10-shot, multiple choice)
12. winograd (0-shot, schema)
13. winogrande (0-shot, schema)
14. bigbench_dyck_languages (10-shot, language modeling)
15. agi_eval_lsat_ar (3-shot, multiple choice)
16. bigbench_cs_algorithms (10-shot, language modeling)
17. bigbench_operators (10-shot, language modeling)
18. bigbench_repeat_copy_logic (10-shot, language modeling)
19. squad (10-shot, language modeling)
20. coqa (0-shot, language modeling)
21. boolq (10-shot, multiple choice)
22. bigbench_language_identification (10-shot, multiple choice)

**Example**:
```
Evaluating: hellaswag_zeroshot (0-shot, type: multiple_choice)... accuracy: 0.5040 | centered: 0.3387 | time: 3.01s
```

**Metrics Available**:
- `accuracy`: Primary metric (0.0 to 1.0)
- `centered`: Centered accuracy (adjusted for random baseline)
- `time`: Execution time in seconds

**Edge Cases**:
- Evaluation sessions span multiple lines (22 consecutive evaluations)
- Should group all evaluations at same step into single marker
- Final evaluation at step 33600 uses different model loading (slower, ~860s total)
- Some evaluations have very short duration (<1s), others longer (>100s)

#### Event Type 3: Checkpoint Events

**Pattern**: `(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*checkpoint_manager.*Saved model parameters.*model_(\d+)\.pt`

**Characteristics**:
- Total occurrences: 9 checkpoint saves (3 distinct checkpoints)
- Each checkpoint consists of 3-4 files:
  - Model parameters (`model_*.pt`)
  - Metadata (`meta_*.json`)
  - Optimizer state per rank (`optim_*_rank0.pt`, `optim_*_rank1.pt`)
- Explicit timestamps in log lines
- Three checkpoint types observed:
  1. `base_checkpoints/d24/model_033600.pt` (base model at step 33600)
  2. `mid_checkpoints/d24/model_000813.pt` (mid-training at step 813)
  3. `chatsft_checkpoints/d24/model_000700.pt` (chat SFT at step 700)

**Example**:
```
2025-12-23 00:54:23,188 - nanochat.checkpoint_manager - INFO - Saved model parameters to: /home/ubuntu/.cache/nanochat/chatsft_checkpoints/d24/model_000700.pt
```

**Timestamp Format**: `YYYY-MM-DD HH:MM:SS,mmm`

**Edge Cases**:
- Multiple log lines per checkpoint (one per file saved)
- Should group all saves for same checkpoint into single marker
- Use "model parameters" line as canonical timestamp
- Checkpoint step numbers don't always match training step (different training phases)

### 2. GPU Metrics CSV Analysis

**File**: `gpu_metrics_v2_20251222_210600.csv`
**Format**: CSV with header row

**Columns**:
1. `timestamp`: Wall-clock time (YYYY-MM-DD HH:MM:SS.mmm)
2. `elapsed_sec`: Seconds since monitoring start (float)
3. `gpu0_power_instant_w`: GPU 0 instantaneous power (Watts)
4. `gpu0_power_avg_w`: GPU 0 average power (Watts)
5. `gpu0_temp_c`: GPU 0 temperature (Celsius)
6. `gpu1_power_instant_w`: GPU 1 instantaneous power (Watts)
7. `gpu1_power_avg_w`: GPU 1 average power (Watts)
8. `gpu1_temp_c`: GPU 1 temperature (Celsius)

**Sample Data**:
```csv
timestamp,elapsed_sec,gpu0_power_instant_w,gpu0_power_avg_w,gpu0_temp_c,gpu1_power_instant_w,gpu1_power_avg_w,gpu1_temp_c
2025-12-22 21:06:00.162,0.000,642.32,686.69,70,675.62,687.30,63
2025-12-22 21:06:00.172,0.010,642.32,686.69,70,675.62,687.30,63
```

**Characteristics**:
- Sampling rate: ~100 Hz (10ms intervals)
- Monitoring start time: `2025-12-22 21:06:00.162`
- Duration: Multiple hours (varies by run)
- File size: Can be very large (millions of rows for long runs)

**Key Insight**:
- GPU metrics use wall-clock timestamps
- Training log also uses wall-clock timestamps
- Can directly correlate by parsing timestamps from both sources

### 3. Timestamp Correlation Strategy

#### Challenge

Two different time representations:
1. **GPU Metrics**: Wall-clock timestamp + elapsed seconds
2. **Training Log**: Wall-clock timestamp (but not on every line)

#### Solution Approach

**Step 1: Extract GPU Monitoring Start Time**
```python
# Read first data row from GPU metrics CSV
gpu_start_time = pd.read_csv(csv_file, nrows=1)['timestamp'].iloc[0]
# Parse: "2025-12-22 21:06:00.162"
gpu_start_dt = datetime.strptime(gpu_start_time, "%Y-%m-%d %H:%M:%S.%f")
```

**Step 2: Parse Training Log Timestamps**
```python
# Training log format: "2025-12-23 00:54:23,188"
# Note: Uses comma for milliseconds, not period
log_timestamp = "2025-12-23 00:54:23,188"
log_dt = datetime.strptime(log_timestamp, "%Y-%m-%d %H:%M:%S,%f")
```

**Step 3: Calculate Elapsed Time**
```python
elapsed_seconds = (log_dt - gpu_start_dt).total_seconds()
elapsed_hours = elapsed_seconds / 3600
```

**Step 4: Handle Missing Timestamps**

For validation events without explicit timestamps:
1. Track last seen timestamp while parsing log
2. Scan backward from validation line to find nearest timestamp
3. Use that timestamp as proxy (acceptable error: <1 second typically)

#### Edge Cases and Challenges

**Challenge 1: Training Started Before GPU Monitoring**
- **Detection**: Training log timestamps before GPU start time
- **Solution**: Filter out events with negative elapsed time
- **Warning**: Notify user that early events are excluded

**Challenge 2: Training Started After GPU Monitoring**
- **Detection**: Gap between GPU start and first training event
- **Solution**: No issue, events will appear after gap
- **Note**: Common when GPU monitoring starts before training script

**Challenge 3: Multiple Training Runs in Same Log**
- **Detection**: Step numbers reset or decrease
- **Solution**: Parse only events within GPU monitoring time window
- **Alternative**: Allow user to specify step range

**Challenge 4: Timestamp Parsing Failures**
- **Detection**: Regex match fails or datetime parsing raises exception
- **Solution**: Skip malformed lines, log warning with line number
- **Robustness**: Continue processing remaining events

**Challenge 5: Evaluation Session Duration**
- **Issue**: 22 evaluations span ~60-90 seconds
- **Solution Option A**: Mark start of evaluation session only
- **Solution Option B**: Mark start and end with shaded region
- **Recommendation**: Option A for simplicity, Option B for detail

### 4. Existing Visualization Architecture

**Current Implementation**: `visualize_gpu_metrics_v2_enhanced.py`

**Key Components**:

1. **Data Loading** (`load_and_process_data`):
   - Chunked CSV reading for memory efficiency
   - Idle period detection using power threshold
   - Automatic cutoff with configurable buffer
   - Downsampling for visualization performance

2. **Plotting** (`plot_metrics`):
   - Plotly subplots (2 rows: power, temperature)
   - Multiple traces per subplot (instant/avg power, GPU 0/1)
   - Custom hover templates with formatted timestamps
   - Vertical line markers (currently only for idle cutoff)

3. **Statistics** (`print_statistics`, `export_summary`):
   - Mean/max/min for all metrics
   - Duration calculation
   - Sample rate estimation

**Vertical Line Implementation** (existing):
```python
fig.add_vline(
    x=cutoff_hours,
    line_dash="dash",
    line_color="red",
    line_width=2,
    opacity=0.7,
    annotation_text=f"Idle Start ({cutoff_hours:.2f}h)",
    annotation_position="top",
    row=row, col=1
)
```

**Key Insight**: Already has infrastructure for adding vertical lines to subplots. Can extend this pattern for training events.

## Design Proposal

### 1. Visual Design

#### Option A: Vertical Lines on Existing Subplots (RECOMMENDED)

**Advantages**:
- Minimal code changes to existing visualization
- Maintains current two-subplot layout
- Events clearly visible on both power and temperature plots
- Interactive legend allows toggling event visibility
- No additional screen real estate required

**Disadvantages**:
- Can become cluttered with many events
- Limited space for detailed annotations
- Difficult to show event-specific metrics (e.g., accuracy values)

**Visual Specification**:

| Event Type | Color | Line Style | Width | Opacity | Annotation Position |
|------------|-------|------------|-------|---------|---------------------|
| Validation | Blue (#1f77b4) | Dotted | 1px | 0.5 | Top |
| Evaluation | Green (#2ca02c) | Dashed | 1.5px | 0.6 | Top |
| Checkpoint | Purple (#9467bd) | Solid | 2px | 0.7 | Top |

**Annotation Format**:
- Validation: `Val: {bpb:.3f}` (e.g., "Val: 0.793")
- Evaluation: `Eval S{step}` (e.g., "Eval S31000")
- Checkpoint: `CP {model_id}` (e.g., "CP 033600")

**Clutter Mitigation**:
- Rotate annotations 90 degrees for dense regions
- Show annotations only on top subplot (power)
- Use annotation font size: 8pt
- Implement annotation overlap detection and adjustment

#### Option B: Three-Subplot Layout with Event Timeline

**Advantages**:
- Dedicated space for event visualization
- Can show detailed metrics (bpb values, accuracy scores)
- Less visual clutter on power/temperature plots
- Easier to see event density and patterns

**Disadvantages**:
- Requires significant refactoring of subplot layout
- Increases total visualization height
- More complex implementation
- May require scrolling on smaller screens

**Layout Specification**:
```
Row 1: Power (instant & average) - 40% height
Row 2: Temperature - 30% height
Row 3: Events timeline - 30% height
```

**Event Timeline Design**:
- Y-axis: Event type (categorical: Validation, Evaluation, Checkpoint)
- X-axis: Time (hours, shared with other subplots)
- Markers: Scatter points with hover details
- Color coding: Same as Option A
- Size: Proportional to importance or duration

#### Recommendation: Option A with Progressive Enhancement

**Phase 1**: Implement Option A (vertical lines)
- Quick to implement
- Immediate value
- Validates timestamp correlation logic

**Phase 2**: Add Option B as alternative view mode
- Command-line flag: `--layout=timeline`
- Allows users to choose based on preference
- Provides detailed view when needed

### 2. Data Structure Design

#### Event Data Model

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class TrainingEvent:
    """Base class for all training events."""
    timestamp: datetime
    elapsed_hours: float
    step: Optional[int]
    event_type: str  # 'validation', 'evaluation', 'checkpoint'

@dataclass
class ValidationEvent(TrainingEvent):
    """Validation event with bpb metric."""
    bpb: float

    def __post_init__(self):
        self.event_type = 'validation'

    def get_label(self) -> str:
        return f"Val: {self.bpb:.3f}"

@dataclass
class EvaluationEvent(TrainingEvent):
    """Evaluation session event."""
    num_tasks: int
    duration_seconds: float
    core_metric: Optional[float]  # CORE metric if available

    def __post_init__(self):
        self.event_type = 'evaluation'

    def get_label(self) -> str:
        if self.core_metric:
            return f"Eval S{self.step} (CORE: {self.core_metric:.3f})"
        return f"Eval S{self.step}"

@dataclass
class CheckpointEvent(TrainingEvent):
    """Checkpoint save event."""
    model_id: str
    checkpoint_type: str  # 'base', 'mid', 'chatsft'

    def __post_init__(self):
        self.event_type = 'checkpoint'

    def get_label(self) -> str:
        return f"CP {self.model_id}"

@dataclass
class EventCollection:
    """Collection of all parsed events."""
    validation_events: List[ValidationEvent]
    evaluation_events: List[EvaluationEvent]
    checkpoint_events: List[CheckpointEvent]

    def get_all_events(self) -> List[TrainingEvent]:
        """Get all events sorted by time."""
        all_events = (
            self.validation_events +
            self.evaluation_events +
            self.checkpoint_events
        )
        return sorted(all_events, key=lambda e: e.elapsed_hours)

    def filter_by_time_range(self, start_hours: float, end_hours: float):
        """Filter events within time range."""
        self.validation_events = [
            e for e in self.validation_events
            if start_hours <= e.elapsed_hours <= end_hours
        ]
        self.evaluation_events = [
            e for e in self.evaluation_events
            if start_hours <= e.elapsed_hours <= end_hours
        ]
        self.checkpoint_events = [
            e for e in self.checkpoint_events
            if start_hours <= e.elapsed_hours <= end_hours
        ]
```

### 3. Parsing Implementation

#### High-Level Algorithm

```
1. Read GPU metrics CSV first row to get monitoring start time
2. Open training log file
3. Initialize event collectors
4. For each line in training log:
   a. Check if line contains timestamp
      - If yes: update current_timestamp
   b. Check if line matches validation pattern
      - If yes: create ValidationEvent with current_timestamp
   c. Check if line matches evaluation pattern
      - If yes: accumulate evaluation info
      - When evaluation session ends: create EvaluationEvent
   d. Check if line matches checkpoint pattern
      - If yes: create CheckpointEvent with line timestamp
5. Calculate elapsed_hours for all events
6. Filter events outside GPU monitoring window
7. Return EventCollection
```

#### Detailed Implementation: Timestamp Tracking

**Challenge**: Not all log lines have timestamps

**Solution**: State machine with timestamp tracking

```python
class LogParser:
    def __init__(self, gpu_start_time: datetime):
        self.gpu_start_time = gpu_start_time
        self.current_timestamp = None
        self.timestamp_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
        )

    def parse_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from line if present."""
        match = self.timestamp_pattern.search(line)
        if match:
            ts_str = match.group(1)
            # Convert comma to period for milliseconds
            ts_str = ts_str.replace(',', '.')
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        return None

    def update_timestamp(self, line: str) -> bool:
        """Update current timestamp if line contains one."""
        ts = self.parse_timestamp(line)
        if ts:
            self.current_timestamp = ts
            return True
        return False

    def get_elapsed_hours(self, timestamp: datetime) -> float:
        """Calculate elapsed hours from GPU monitoring start."""
        elapsed_sec = (timestamp - self.gpu_start_time).total_seconds()
        return elapsed_sec / 3600
```

#### Detailed Implementation: Validation Event Parsing

```python
def parse_validation_events(
    log_file: str,
    parser: LogParser
) -> List[ValidationEvent]:
    """Parse validation events from training log."""
    events = []
    validation_pattern = re.compile(
        r'Step (\d+) \| Validation bpb: ([\d.]+)'
    )

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Update timestamp tracker
            parser.update_timestamp(line)

            # Check for validation event
            match = validation_pattern.search(line)
            if match:
                step = int(match.group(1))
                bpb = float(match.group(2))

                if parser.current_timestamp is None:
                    print(f"Warning: No timestamp for validation at line {line_num}")
                    continue

                elapsed_hours = parser.get_elapsed_hours(parser.current_timestamp)

                # Filter events outside GPU monitoring window
                if elapsed_hours < 0:
                    continue

                event = ValidationEvent(
                    timestamp=parser.current_timestamp,
                    elapsed_hours=elapsed_hours,
                    step=step,
                    bpb=bpb,
                    event_type='validation'
                )
                events.append(event)

    return events
```

#### Detailed Implementation: Evaluation Event Parsing

**Challenge**: Each evaluation session consists of 22 consecutive lines

**Solution**: Session detection and aggregation

```python
def parse_evaluation_events(
    log_file: str,
    parser: LogParser
) -> List[EvaluationEvent]:
    """Parse evaluation session events from training log."""
    events = []
    eval_pattern = re.compile(r'Evaluating: (\w+)')
    core_pattern = re.compile(r'Step (\d+) \| CORE metric: ([\d.]+)')

    current_session = None
    session_start_time = None
    session_start_line = None
    task_count = 0

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parser.update_timestamp(line)

            # Check for evaluation task
            eval_match = eval_pattern.search(line)
            if eval_match:
                if current_session is None:
                    # Start of new evaluation session
                    current_session = eval_match.group(1)
                    session_start_time = parser.current_timestamp
                    session_start_line = line_num
                    task_count = 1
                else:
                    # Continuation of current session
                    task_count += 1

            # Check for CORE metric (end of evaluation session)
            core_match = core_pattern.search(line)
            if core_match and current_session is not None:
                step = int(core_match.group(1))
                core_metric = float(core_match.group(2))

                if session_start_time is None:
                    print(f"Warning: No timestamp for evaluation session at line {line_num}")
                    current_session = None
                    continue

                # Calculate session duration
                duration_sec = (parser.current_timestamp - session_start_time).total_seconds()
                elapsed_hours = parser.get_elapsed_hours(session_start_time)

                # Filter events outside GPU monitoring window
                if elapsed_hours >= 0:
                    event = EvaluationEvent(
                        timestamp=session_start_time,
                        elapsed_hours=elapsed_hours,
                        step=step,
                        num_tasks=task_count,
                        duration_seconds=duration_sec,
                        core_metric=core_metric,
                        event_type='evaluation'
                    )
                    events.append(event)

                # Reset session tracking
                current_session = None
                session_start_time = None
                task_count = 0

    return events
```

#### Detailed Implementation: Checkpoint Event Parsing

```python
def parse_checkpoint_events(
    log_file: str,
    parser: LogParser
) -> List[CheckpointEvent]:
    """Parse checkpoint save events from training log."""
    events = []
    checkpoint_pattern = re.compile(
        r'checkpoint_manager.*Saved model parameters.*'
        r'(\w+)_checkpoints/d24/model_(\d+)\.pt'
    )

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Checkpoint lines always have timestamps
            timestamp = parser.parse_timestamp(line)
            if timestamp:
                parser.current_timestamp = timestamp

            # Check for checkpoint save
            match = checkpoint_pattern.search(line)
            if match:
                checkpoint_type = match.group(1)  # 'base', 'mid', 'chatsft'
                model_id = match.group(2)

                if parser.current_timestamp is None:
                    print(f"Warning: No timestamp for checkpoint at line {line_num}")
                    continue

                elapsed_hours = parser.get_elapsed_hours(parser.current_timestamp)

                # Filter events outside GPU monitoring window
                if elapsed_hours < 0:
                    continue

                event = CheckpointEvent(
                    timestamp=parser.current_timestamp,
                    elapsed_hours=elapsed_hours,
                    step=None,  # Step not directly available in checkpoint line
                    model_id=model_id,
                    checkpoint_type=checkpoint_type,
                    event_type='checkpoint'
                )
                events.append(event)

    return events
```

#### Main Parsing Function

```python
def parse_training_log(
    log_file: str,
    gpu_start_time: datetime
) -> EventCollection:
    """
    Parse all training events from log file.

    Args:
        log_file: Path to training log file
        gpu_start_time: GPU monitoring start time for correlation

    Returns:
        EventCollection with all parsed events
    """
    print(f"Parsing training log: {log_file}")
    print(f"GPU monitoring start: {gpu_start_time}")

    parser = LogParser(gpu_start_time)

    # Parse each event type
    validation_events = parse_validation_events(log_file, parser)
    evaluation_events = parse_evaluation_events(log_file, parser)
    checkpoint_events = parse_checkpoint_events(log_file, parser)

    print(f"Parsed {len(validation_events)} validation events")
    print(f"Parsed {len(evaluation_events)} evaluation events")
    print(f"Parsed {len(checkpoint_events)} checkpoint events")

    collection = EventCollection(
        validation_events=validation_events,
        evaluation_events=evaluation_events,
        checkpoint_events=checkpoint_events
    )

    return collection
```

### 4. Visualization Integration

#### Modified plot_metrics Function Signature

```python
def plot_metrics(
    df: pd.DataFrame,
    cutoff_time: Optional[float] = None,
    events: Optional[EventCollection] = None,
    output_file: str = 'gpu_metrics_plot.html',
    show_validation: bool = True,
    show_evaluation: bool = True,
    show_checkpoint: bool = True
) -> go.Figure:
    """
    Create interactive visualization with optional training events.

    Args:
        df: GPU metrics dataframe
        cutoff_time: Idle cutoff time in seconds (optional)
        events: Training events to overlay (optional)
        output_file: Output HTML file path
        show_validation: Whether to show validation events
        show_evaluation: Whether to show evaluation events
        show_checkpoint: Whether to show checkpoint events

    Returns:
        Plotly figure object
    """
```

#### Event Overlay Implementation

```python
def add_event_markers(
    fig: go.Figure,
    events: EventCollection,
    show_validation: bool = True,
    show_evaluation: bool = True,
    show_checkpoint: bool = True
) -> None:
    """Add training event markers to existing figure."""

    # Event visual specifications
    event_specs = {
        'validation': {
            'color': '#1f77b4',
            'dash': 'dot',
            'width': 1,
            'opacity': 0.5
        },
        'evaluation': {
            'color': '#2ca02c',
            'dash': 'dash',
            'width': 1.5,
            'opacity': 0.6
        },
        'checkpoint': {
            'color': '#9467bd',
            'dash': 'solid',
            'width': 2,
            'opacity': 0.7
        }
    }

    # Add validation events
    if show_validation:
        for event in events.validation_events:
            for row in [1, 2]:  # Add to both subplots
                fig.add_vline(
                    x=event.elapsed_hours,
                    line_dash=event_specs['validation']['dash'],
                    line_color=event_specs['validation']['color'],
                    line_width=event_specs['validation']['width'],
                    opacity=event_specs['validation']['opacity'],
                    annotation_text=event.get_label(),
                    annotation_position="top" if row == 1 else "bottom",
                    annotation_font_size=8,
                    annotation_textangle=-90,
                    row=row, col=1
                )

    # Add evaluation events
    if show_evaluation:
        for event in events.evaluation_events:
            for row in [1, 2]:
                fig.add_vline(
                    x=event.elapsed_hours,
                    line_dash=event_specs['evaluation']['dash'],
                    line_color=event_specs['evaluation']['color'],
                    line_width=event_specs['evaluation']['width'],
                    opacity=event_specs['evaluation']['opacity'],
                    annotation_text=event.get_label(),
                    annotation_position="top" if row == 1 else "bottom",
                    annotation_font_size=8,
                    annotation_textangle=-90,
                    row=row, col=1
                )

    # Add checkpoint events
    if show_checkpoint:
        for event in events.checkpoint_events:
            for row in [1, 2]:
                fig.add_vline(
                    x=event.elapsed_hours,
                    line_dash=event_specs['checkpoint']['dash'],
                    line_color=event_specs['checkpoint']['color'],
                    line_width=event_specs['checkpoint']['width'],
                    opacity=event_specs['checkpoint']['opacity'],
                    annotation_text=event.get_label(),
                    annotation_position="top" if row == 1 else "bottom",
                    annotation_font_size=8,
                    annotation_textangle=-90,
                    row=row, col=1
                )
```

### 5. Command-Line Interface

#### Enhanced Argument Parser

```python
def create_argument_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser with training log support."""
    parser = argparse.ArgumentParser(
        description='Visualize GPU metrics with optional training events overlay.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (no events)
  python visualize_gpu_metrics_v2_with_events.py gpu_metrics_v2_20251222_210600.csv

  # With training events
  python visualize_gpu_metrics_v2_with_events.py \\
      gpu_metrics_v2_20251222_210600.csv \\
      --training-log nanochat/training_20251221_183935.log

  # With custom downsampling and event filtering
  python visualize_gpu_metrics_v2_with_events.py \\
      gpu_metrics_v2_20251222_210600.csv \\
      --training-log nanochat/training_20251221_183935.log \\
      --downsample-factor 50 \\
      --no-validation \\
      --show-checkpoint-only

  # Export events to CSV for analysis
  python visualize_gpu_metrics_v2_with_events.py \\
      gpu_metrics_v2_20251222_210600.csv \\
      --training-log nanochat/training_20251221_183935.log \\
      --export-events events.csv
        """
    )

    # Required arguments
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to GPU metrics CSV file'
    )

    # Training log arguments
    parser.add_argument(
        '--training-log',
        type=str,
        default=None,
        help='Path to training log file (optional)'
    )

    # Visualization control
    parser.add_argument(
        '--downsample-factor',
        type=int,
        default=10,
        help='Downsampling factor for GPU metrics (default: 10)'
    )

    parser.add_argument(
        '--power-threshold',
        type=float,
        default=200,
        help='Power threshold in Watts for idle detection (default: 200)'
    )

    parser.add_argument(
        '--buffer-minutes',
        type=int,
        default=10,
        help='Minutes to add after idle cutoff (default: 10)'
    )

    # Event filtering
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Hide validation events'
    )

    parser.add_argument(
        '--no-evaluation',
        action='store_true',
        help='Hide evaluation events'
    )

    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Hide checkpoint events'
    )

    parser.add_argument(
        '--show-validation-only',
        action='store_true',
        help='Show only validation events'
    )

    parser.add_argument(
        '--show-evaluation-only',
        action='store_true',
        help='Show only evaluation events'
    )

    parser.add_argument(
        '--show-checkpoint-only',
        action='store_true',
        help='Show only checkpoint events'
    )

    # Export options
    parser.add_argument(
        '--export-events',
        type=str,
        default=None,
        help='Export parsed events to CSV file (optional)'
    )

    return parser
```

### 6. Error Handling and Validation

#### Robust File Handling

```python
def validate_inputs(args) -> Tuple[bool, Optional[str]]:
    """
    Validate input files and arguments.

    Returns:
        (is_valid, error_message)
    """
    # Check GPU metrics CSV exists
    if not os.path.exists(args.csv_file):
        return False, f"GPU metrics CSV not found: {args.csv_file}"

    # Check training log exists (if provided)
    if args.training_log and not os.path.exists(args.training_log):
        return False, f"Training log not found: {args.training_log}"

    # Validate downsample factor
    if args.downsample_factor < 1:
        return False, "Downsample factor must be >= 1"

    # Validate power threshold
    if args.power_threshold < 0:
        return False, "Power threshold must be >= 0"

    # Check for conflicting event filter flags
    exclusive_flags = [
        args.show_validation_only,
        args.show_evaluation_only,
        args.show_checkpoint_only
    ]
    if sum(exclusive_flags) > 1:
        return False, "Cannot use multiple --show-*-only flags together"

    return True, None
```

#### Graceful Degradation

```python
def safe_parse_training_log(
    log_file: str,
    gpu_start_time: datetime
) -> Optional[EventCollection]:
    """
    Parse training log with error handling.
    Returns None if parsing fails completely.
    """
    try:
        events = parse_training_log(log_file, gpu_start_time)

        # Warn if no events found
        total_events = (
            len(events.validation_events) +
            len(events.evaluation_events) +
            len(events.checkpoint_events)
        )

        if total_events == 0:
            print("Warning: No training events found in log file")
            print("Possible reasons:")
            print("  - Training started before GPU monitoring")
            print("  - Training log format has changed")
            print("  - Log file is from different training run")
            return None

        return events

    except Exception as e:
        print(f"Error parsing training log: {e}")
        print("Continuing without training events...")
        import traceback
        traceback.print_exc()
        return None
```

## Implementation Plan

### Phase 1: Core Functionality (Week 1)

#### Step 1.1: Create New Script File
- **File**: `visualize_gpu_metrics_v2_with_events.py`
- **Action**: Copy from `visualize_gpu_metrics_v2_enhanced.py`
- **Rationale**: Preserve existing working script, allow parallel development

#### Step 1.2: Implement Data Models
- **Components**:
  - `TrainingEvent` base class
  - `ValidationEvent`, `EvaluationEvent`, `CheckpointEvent` subclasses
  - `EventCollection` container
- **Testing**: Unit tests for data model creation and methods
- **Validation**: Ensure proper inheritance and type hints

#### Step 1.3: Implement Log Parser
- **Components**:
  - `LogParser` class with timestamp tracking
  - `parse_validation_events()` function
  - `parse_evaluation_events()` function
  - `parse_checkpoint_events()` function
  - `parse_training_log()` main function
- **Testing**:
  - Test with sample log snippets
  - Test timestamp correlation accuracy
  - Test edge cases (missing timestamps, malformed lines)
- **Validation**: Compare parsed event counts with manual inspection

#### Step 1.4: Integrate Event Overlay
- **Components**:
  - Modify `plot_metrics()` to accept `EventCollection`
  - Implement `add_event_markers()` function
  - Add event visibility controls
- **Testing**:
  - Visual inspection of generated plots
  - Verify event positions match log timestamps
  - Test with different event filter combinations
- **Validation**: Cross-reference event times with log file

#### Step 1.5: Command-Line Interface
- **Components**:
  - Enhanced argument parser
  - Input validation
  - Error handling
- **Testing**:
  - Test all argument combinations
  - Test error messages for invalid inputs
  - Test help text clarity
- **Validation**: User acceptance testing

### Phase 2: Refinement and Optimization (Week 2)

#### Step 2.1: Performance Optimization
- **Issues to Address**:
  - Large log files (40K+ lines) parsing time
  - Memory usage with many events
  - Rendering performance with dense event markers
- **Solutions**:
  - Implement streaming log parser (process line-by-line)
  - Add event density-based annotation thinning
  - Optimize regex patterns for speed
- **Metrics**: Measure parsing time and memory usage

#### Step 2.2: Visual Refinement
- **Enhancements**:
  - Implement annotation overlap detection
  - Add annotation rotation for dense regions
  - Improve color contrast and accessibility
  - Add event type legend
- **Testing**: Visual inspection with various event densities
- **Validation**: User feedback on readability

#### Step 2.3: Event Export Functionality
- **Feature**: Export parsed events to CSV
- **Format**:
  ```csv
  event_type,timestamp,elapsed_hours,step,metric_name,metric_value
  validation,2025-12-22 22:30:15.123,1.234,2000,bpb,0.9326
  evaluation,2025-12-22 22:31:45.678,1.260,2000,core,0.1300
  checkpoint,2025-12-23 00:54:23.188,3.805,33600,model_id,033600
  ```
- **Use Cases**:
  - Further analysis in spreadsheet tools
  - Integration with other visualization tools
  - Debugging timestamp correlation
- **Testing**: Verify CSV format and data accuracy

### Phase 3: Advanced Features (Week 3)

#### Step 3.1: Timeline Subplot (Option B)
- **Implementation**:
  - Add `--layout=timeline` flag
  - Create three-subplot layout
  - Implement event timeline visualization
- **Testing**: Compare with vertical line approach
- **Validation**: User preference survey

#### Step 3.2: Interactive Event Details
- **Feature**: Enhanced hover information for events
- **Implementation**:
  - Add invisible scatter traces at event positions
  - Custom hover templates with full event details
  - Link to log file line numbers
- **Example Hover**:
  ```
  Validation Event
  Step: 31000
  BPB: 0.7926
  Time: 2025-12-22 23:45:12
  Elapsed: 2.65 hours
  ```

#### Step 3.3: Event Correlation Analysis
- **Feature**: Automatic correlation between events and GPU metrics
- **Metrics**:
  - Average power during evaluation sessions
  - Temperature spike detection at checkpoints
  - Power efficiency per validation improvement
- **Output**: Summary statistics in console and exported file

### Phase 4: Documentation and Testing (Week 4)

#### Step 4.1: Comprehensive Testing
- **Unit Tests**:
  - Test each parsing function independently
  - Test timestamp correlation edge cases
  - Test event filtering logic
- **Integration Tests**:
  - End-to-end test with real log files
  - Test with various GPU metrics CSV formats
  - Test error handling paths
- **Performance Tests**:
  - Benchmark with large log files (100K+ lines)
  - Memory profiling
  - Rendering performance measurement

#### Step 4.2: Documentation
- **User Guide**:
  - Installation instructions
  - Usage examples
  - Troubleshooting guide
- **Developer Guide**:
  - Code architecture overview
  - Adding new event types
  - Extending visualization options
- **API Documentation**:
  - Docstrings for all public functions
  - Type hints for all parameters
  - Example code snippets

#### Step 4.3: Example Gallery
- **Create**:
  - Sample visualizations with different configurations
  - Annotated screenshots explaining features
  - Comparison before/after event overlay
- **Purpose**:
  - Help users understand capabilities
  - Provide visual reference for expected output
  - Demonstrate best practices

## Edge Cases and Challenges

### Challenge 1: Timestamp Misalignment

**Scenario**: GPU monitoring started significantly before or after training

**Detection**:
- Check if first/last training event is within GPU metrics time range
- Calculate percentage of events outside range

**Mitigation**:
- Display warning with time offset information
- Suggest correct GPU metrics CSV file
- Allow manual time offset adjustment via CLI flag

**Example**:
```
Warning: Training log time range mismatch
  GPU monitoring: 2025-12-22 21:06:00 to 2025-12-23 03:15:30
  Training events: 2025-12-22 18:39:35 to 2025-12-23 00:54:23
  Events before GPU start: 143 (100%)

Suggestion: Use GPU metrics CSV from earlier monitoring session
```

### Challenge 2: Multiple Training Phases

**Scenario**: Log contains multiple training phases (pretraining, mid-training, SFT)

**Detection**:
- Step numbers reset or jump discontinuously
- Different checkpoint types appear
- Multiple "Starting training" messages

**Mitigation**:
- Parse all phases, label by checkpoint type
- Add phase boundaries as distinct markers
- Allow filtering by training phase via CLI

**Implementation**:
```python
def detect_training_phases(events: EventCollection) -> List[TrainingPhase]:
    """Detect distinct training phases from events."""
    phases = []
    current_phase = None

    for event in sorted(events.get_all_events(), key=lambda e: e.elapsed_hours):
        if isinstance(event, CheckpointEvent):
            if current_phase is None or event.checkpoint_type != current_phase.type:
                # New phase detected
                current_phase = TrainingPhase(
                    type=event.checkpoint_type,
                    start_time=event.elapsed_hours
                )
                phases.append(current_phase)

    return phases
```

### Challenge 3: Log Format Variations

**Scenario**: Different nanochat versions use different log formats

**Detection**:
- Regex patterns fail to match
- Unexpected field formats
- Missing expected fields

**Mitigation**:
- Implement multiple regex patterns per event type
- Try patterns in order of specificity
- Log which pattern matched for debugging
- Provide pattern override via config file

**Example**:
```python
VALIDATION_PATTERNS = [
    # Current format
    r'Step (\d+) \| Validation bpb: ([\d.]+)',
    # Legacy format (hypothetical)
    r'Validation at step (\d+): bpb=([\d.]+)',
    # Alternative format
    r'\[Step (\d+)\] Val BPB: ([\d.]+)'
]

def try_parse_validation(line: str) -> Optional[Tuple[int, float]]:
    """Try multiple patterns to parse validation event."""
    for pattern in VALIDATION_PATTERNS:
        match = re.search(pattern, line)
        if match:
            return int(match.group(1)), float(match.group(2))
    return None
```

### Challenge 4: Annotation Overlap

**Scenario**: Many events occur close together, annotations overlap

**Detection**:
- Calculate annotation bounding boxes
- Check for overlaps in x-axis position

**Mitigation Strategy 1**: Stagger annotations vertically
```python
def stagger_annotations(events: List[TrainingEvent], min_spacing_hours: float = 0.1):
    """Adjust annotation positions to avoid overlap."""
    sorted_events = sorted(events, key=lambda e: e.elapsed_hours)

    for i in range(1, len(sorted_events)):
        prev_event = sorted_events[i-1]
        curr_event = sorted_events[i]

        if curr_event.elapsed_hours - prev_event.elapsed_hours < min_spacing_hours:
            # Alternate annotation position
            curr_event.annotation_position = (
                'bottom' if prev_event.annotation_position == 'top' else 'top'
            )
```

**Mitigation Strategy 2**: Reduce annotation density
```python
def thin_annotations(events: List[TrainingEvent], max_density: int = 20):
    """Show only subset of annotations in dense regions."""
    if len(events) <= max_density:
        return events

    # Keep every Nth event
    step = len(events) // max_density
    return events[::step]
```

**Mitigation Strategy 3**: Aggregate nearby events
```python
def aggregate_nearby_events(
    events: List[TrainingEvent],
    time_window_hours: float = 0.05
) -> List[EventGroup]:
    """Group events that occur within time window."""
    groups = []
    current_group = []

    for event in sorted(events, key=lambda e: e.elapsed_hours):
        if not current_group:
            current_group.append(event)
        else:
            time_diff = event.elapsed_hours - current_group[-1].elapsed_hours
            if time_diff <= time_window_hours:
                current_group.append(event)
            else:
                groups.append(EventGroup(current_group))
                current_group = [event]

    if current_group:
        groups.append(EventGroup(current_group))

    return groups
```

### Challenge 5: Large Log Files

**Scenario**: Training logs exceed 100K lines, slow parsing

**Current Performance** (estimated):
- 40K lines: ~2-3 seconds
- 100K lines: ~5-8 seconds
- 500K lines: ~25-40 seconds

**Optimization 1**: Streaming parser
```python
def parse_training_log_streaming(
    log_file: str,
    gpu_start_time: datetime
) -> EventCollection:
    """Memory-efficient streaming parser."""
    events = EventCollection([], [], [])

    with open(log_file, 'r') as f:
        for line in f:
            # Process one line at a time
            # Don't load entire file into memory
            event = parse_line(line, gpu_start_time)
            if event:
                events.add_event(event)

    return events
```

**Optimization 2**: Compiled regex patterns
```python
# Compile patterns once at module level
VALIDATION_PATTERN = re.compile(r'Step (\d+) \| Validation bpb: ([\d.]+)')
EVAL_PATTERN = re.compile(r'Evaluating: (\w+)')
CHECKPOINT_PATTERN = re.compile(r'checkpoint_manager.*Saved model parameters')

# Use compiled patterns in parsing
match = VALIDATION_PATTERN.search(line)  # Faster than re.search()
```

**Optimization 3**: Early termination
```python
def parse_training_log_with_cutoff(
    log_file: str,
    gpu_start_time: datetime,
    gpu_end_time: datetime
) -> EventCollection:
    """Stop parsing after GPU monitoring ends."""
    events = EventCollection([], [], [])

    with open(log_file, 'r') as f:
        for line in f:
            timestamp = parse_timestamp(line)
            if timestamp and timestamp > gpu_end_time:
                # No more events will be in range
                break

            event = parse_line(line, gpu_start_time)
            if event:
                events.add_event(event)

    return events
```

### Challenge 6: Missing or Corrupted Timestamps

**Scenario**: Log lines missing timestamps due to buffering or errors

**Detection**:
- Timestamp parsing returns None
- Large gaps between consecutive timestamps
- Timestamps out of chronological order

**Mitigation**:
- Interpolate timestamps based on surrounding lines
- Use line number as fallback ordering mechanism
- Warn user about timestamp quality issues

**Implementation**:
```python
def interpolate_timestamp(
    prev_timestamp: datetime,
    next_timestamp: datetime,
    prev_line: int,
    curr_line: int,
    next_line: int
) -> datetime:
    """Interpolate timestamp for line without explicit timestamp."""
    # Linear interpolation based on line numbers
    total_lines = next_line - prev_line
    curr_offset = curr_line - prev_line
    fraction = curr_offset / total_lines

    time_diff = (next_timestamp - prev_timestamp).total_seconds()
    interpolated_offset = time_diff * fraction

    return prev_timestamp + timedelta(seconds=interpolated_offset)
```

## Testing Strategy

### Unit Tests

**Test File**: `test_event_parsing.py`

```python
import unittest
from datetime import datetime
from visualize_gpu_metrics_v2_with_events import (
    LogParser, parse_validation_events, ValidationEvent
)

class TestLogParser(unittest.TestCase):
    def setUp(self):
        self.gpu_start = datetime(2025, 12, 22, 21, 6, 0, 162000)
        self.parser = LogParser(self.gpu_start)

    def test_parse_timestamp_standard_format(self):
        line = "2025-12-22 23:45:12,345 - INFO - Test message"
        ts = self.parser.parse_timestamp(line)
        self.assertIsNotNone(ts)
        self.assertEqual(ts.year, 2025)
        self.assertEqual(ts.month, 12)
        self.assertEqual(ts.day, 22)
        self.assertEqual(ts.hour, 23)
        self.assertEqual(ts.minute, 45)
        self.assertEqual(ts.second, 12)

    def test_parse_timestamp_no_timestamp(self):
        line = "This line has no timestamp"
        ts = self.parser.parse_timestamp(line)
        self.assertIsNone(ts)

    def test_elapsed_hours_calculation(self):
        # 2 hours 39 minutes 12 seconds after GPU start
        event_time = datetime(2025, 12, 22, 23, 45, 12, 345000)
        elapsed = self.parser.get_elapsed_hours(event_time)
        expected = 2.653  # Approximately
        self.assertAlmostEqual(elapsed, expected, places=2)

    def test_validation_event_parsing(self):
        # Create temporary test log file
        test_log = "test_validation.log"
        with open(test_log, 'w') as f:
            f.write("2025-12-22 22:30:15,123 - INFO - Starting\n")
            f.write("Step 2000 | Validation bpb: 0.9326\n")
            f.write("2025-12-22 23:45:12,345 - INFO - Continuing\n")
            f.write("Step 4000 | Validation bpb: 0.8884\n")

        events = parse_validation_events(test_log, self.parser)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].step, 2000)
        self.assertAlmostEqual(events[0].bpb, 0.9326)
        self.assertEqual(events[1].step, 4000)
        self.assertAlmostEqual(events[1].bpb, 0.8884)

        # Cleanup
        os.remove(test_log)

class TestEventCollection(unittest.TestCase):
    def test_filter_by_time_range(self):
        # Create test events
        events = EventCollection(
            validation_events=[
                ValidationEvent(datetime.now(), 1.0, 1000, 0.95, 'validation'),
                ValidationEvent(datetime.now(), 2.0, 2000, 0.90, 'validation'),
                ValidationEvent(datetime.now(), 3.0, 3000, 0.85, 'validation'),
            ],
            evaluation_events=[],
            checkpoint_events=[]
        )

        # Filter to 1.5-2.5 hour range
        events.filter_by_time_range(1.5, 2.5)

        self.assertEqual(len(events.validation_events), 1)
        self.assertEqual(events.validation_events[0].step, 2000)
```

### Integration Tests

**Test File**: `test_integration.py`

```python
def test_end_to_end_visualization():
    """Test complete workflow from CSV + log to HTML output."""
    # Use real sample files
    csv_file = "test_data/sample_gpu_metrics.csv"
    log_file = "test_data/sample_training.log"
    output_file = "test_output.html"

    # Run visualization
    main([
        csv_file,
        '--training-log', log_file,
        '--downsample-factor', '1',
        '--output', output_file
    ])

    # Verify output exists
    assert os.path.exists(output_file)

    # Verify output contains expected elements
    with open(output_file, 'r') as f:
        content = f.read()
        assert 'GPU Metrics Visualization' in content
        assert 'Validation' in content or len(validation_events) == 0
        assert 'Evaluation' in content or len(evaluation_events) == 0

    # Cleanup
    os.remove(output_file)
```

### Performance Tests

**Test File**: `test_performance.py`

```python
import time
import memory_profiler

def test_parsing_performance():
    """Measure parsing time for various log sizes."""
    log_sizes = [1000, 10000, 50000, 100000]

    for size in log_sizes:
        # Generate test log
        test_log = generate_test_log(size)

        # Measure parsing time
        start = time.time()
        events = parse_training_log(test_log, datetime.now())
        elapsed = time.time() - start

        print(f"Log size: {size} lines")
        print(f"Parse time: {elapsed:.2f}s")
        print(f"Rate: {size/elapsed:.0f} lines/sec")
        print()

        # Cleanup
        os.remove(test_log)

@memory_profiler.profile
def test_memory_usage():
    """Profile memory usage during parsing."""
    large_log = generate_test_log(100000)
    events = parse_training_log(large_log, datetime.now())
    os.remove(large_log)
```

## Success Criteria

### Functional Requirements

1. **Parsing Accuracy**: 100% of valid events correctly parsed
2. **Timestamp Correlation**: <1 second error in event positioning
3. **Visualization Quality**: Events clearly visible and distinguishable
4. **Performance**: Parse 40K line log in <5 seconds
5. **Robustness**: Handle malformed logs without crashing

### User Experience Requirements

1. **Ease of Use**: Single command to generate visualization with events
2. **Clarity**: Event annotations readable without zooming
3. **Interactivity**: Hover shows full event details
4. **Flexibility**: Can show/hide event types independently
5. **Documentation**: Clear examples and troubleshooting guide

### Technical Requirements

1. **Code Quality**: Type hints, docstrings, PEP 8 compliance
2. **Test Coverage**: >80% code coverage
3. **Maintainability**: Modular design, easy to extend
4. **Compatibility**: Works with existing GPU metrics CSV format
5. **Error Handling**: Graceful degradation on parsing errors

## Future Enhancements

### Enhancement 1: Real-Time Monitoring

**Feature**: Live updating visualization during training

**Implementation**:
- Watch log file for changes
- Incrementally parse new events
- Update visualization in browser via WebSocket
- Refresh every N seconds

**Use Case**: Monitor training progress in real-time

### Enhancement 2: Multi-Run Comparison

**Feature**: Overlay events from multiple training runs

**Implementation**:
- Accept multiple log files
- Color-code events by run
- Align by step number or elapsed time
- Show comparative metrics

**Use Case**: Compare training efficiency across experiments

### Enhancement 3: Anomaly Detection

**Feature**: Automatically detect unusual patterns

**Detection Criteria**:
- Validation bpb increases (regression)
- Evaluation time significantly longer than average
- Checkpoint save failures
- GPU power spikes during evaluation

**Output**: Highlighted markers with warning annotations

### Enhancement 4: Export to TensorBoard

**Feature**: Convert events to TensorBoard format

**Implementation**:
- Generate TensorBoard event files
- Include GPU metrics as custom scalars
- Link training events to TensorBoard timeline

**Use Case**: Integration with existing ML workflow tools

### Enhancement 5: Configuration Profiles

**Feature**: Save/load visualization preferences

**Format**: YAML configuration file
```yaml
visualization:
  downsample_factor: 10
  power_threshold: 200

events:
  show_validation: true
  show_evaluation: true
  show_checkpoint: false
  annotation_font_size: 8

colors:
  validation: "#1f77b4"
  evaluation: "#2ca02c"
  checkpoint: "#9467bd"
```

**Use Case**: Consistent visualization across team members

## Conclusion

This DevLog provides a comprehensive plan for integrating training events into GPU metrics visualization. The proposed solution:

1. **Preserves existing functionality** while adding new capabilities
2. **Handles edge cases** robustly with graceful degradation
3. **Provides clear implementation path** with phased approach
4. **Includes thorough testing strategy** for reliability
5. **Considers future extensibility** for additional features

The implementation will significantly enhance the ability to correlate training progress with GPU resource utilization, enabling better optimization and debugging of LLM training workflows.

## Next Steps

### RESOLVED: Console Logging Timestamp Solution

**Issue**: Training logs lacked consistent timestamps because output came from multiple sources (shell scripts, package managers, Python subprocesses, Rust builds) that `print0()` modifications could not address.

**Solution Implemented**: Added timestamp prepending at the outermost shell redirection layer using `awk` to capture ALL output sources uniformly. The training script is now invoked with `./run_2gpu_d24.sh 2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }' | tee training_(date +%Y%m%d_%H%M%S).log` which prepends `YYYY-MM-DD HH:MM:SS` to every line regardless of origin, with immediate flushing for real-time visibility. This approach is superior to modifying individual Python functions because it provides universal coverage across all subprocess outputs (uv, maturin, torchrun, dataset downloads, tokenizer training) and requires no code changes to the training codebase itself.

**Immediate Fix Required**:
1. Update `print0()` to use Python's logging module with timestamps
2. Update all training scripts to use timestamped logging
3. This will make ALL console logs parseable without regex hacks

**Implementation Priority**:
1. **FIRST**: Fix `print0()` to add timestamps (this DevLog section)
2. **SECOND**: Implement visualization with proper timestamp parsing
3. **THIRD**: Consider structured event logger for future enhancements

### Implementation Steps

1. **Fix logging infrastructure** (IMMEDIATE - this week)
   - Update `print0()` in `nanochat/common.py`
   - Test with existing training scripts
   - Verify timestamps appear in console output

2. **Update visualization script** (Week 2)
   - Simplify parsing logic (no timestamp interpolation needed)
   - Use consistent timestamp format
   - Test with new timestamped logs

3. **Optional enhancements** (Week 3-4)
   - Add structured event logger (JSON Lines)
   - Create event export functionality
   - Implement timeline visualization

---

**Document Status**: Implementation Plan Updated - Fix Logging First
**Estimated Effort**: 1 week for logging fix + 2-3 weeks for visualization
**Dependencies**: None (logging fix is self-contained)
**Risk Level**: Low (backward compatible, additive changes)
**Priority**: HIGH - Logging fix should be done immediately


