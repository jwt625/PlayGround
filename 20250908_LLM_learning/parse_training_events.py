#!/usr/bin/env python3
"""
Parse training log file and extract significant events with timestamps.

Events extracted:
- Validation (bpb and loss)
- CORE metric evaluations
- Benchmark evaluations (Evaluating: ...)
- Checkpoint saves
- Model loading
- Accuracy results (ARC-Easy, MMLU, HumanEval)
- Inline benchmark results (mmlu_acc, arc_easy_acc)

Usage:
    python parse_training_events.py <log_file> [--output events.csv]
"""

import re
import csv
import argparse
from datetime import datetime
from pathlib import Path


EVENT_PATTERNS = [
    # Validation bpb (base training)
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Step (\d+) \| Validation bpb: ([\d.]+)',
     'validation_bpb', lambda m: {'step': int(m.group(2)), 'value': float(m.group(3))}),
    
    # Validation loss (SFT)
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Step (\d+) \| Validation loss: ([\d.]+)',
     'validation_loss', lambda m: {'step': int(m.group(2)), 'value': float(m.group(3))}),
    
    # CORE metric
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Step (\d+) \| CORE metric: ([\d.]+)',
     'core_metric', lambda m: {'step': int(m.group(2)), 'value': float(m.group(3))}),
    
    # Inline benchmark (mmlu_acc, arc_easy_acc)
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Step (\d+) \| mmlu_acc: ([\d.]+), arc_easy_acc: ([\d.]+)',
     'inline_benchmark', lambda m: {'step': int(m.group(2)), 'mmlu': float(m.group(3)), 'arc_easy': float(m.group(4))}),
    
    # Individual benchmark evaluation
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Evaluating: (\S+) \((\d+)-shot.*accuracy: ([\d.]+)',
     'benchmark_eval', lambda m: {'benchmark': m.group(2), 'shots': int(m.group(3)), 'accuracy': float(m.group(4))}),
    
    # Checkpoint save
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Saved model parameters to: (.+)',
     'checkpoint_save', lambda m: {'path': m.group(2)}),
    
    # Model loading
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Loading model from (.+) with step (\d+)',
     'model_load', lambda m: {'path': m.group(2), 'step': int(m.group(3))}),
    
    # ARC-Easy accuracy
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*ARC-Easy accuracy: ([\d.]+)%',
     'arc_easy_result', lambda m: {'accuracy': float(m.group(2))}),
    
    # MMLU accuracy
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*MMLU accuracy: ([\d.]+)%',
     'mmlu_result', lambda m: {'accuracy': float(m.group(2))}),
    
    # HumanEval accuracy
    (r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*HumanEval accuracy: ([\d.]+)%',
     'humaneval_result', lambda m: {'accuracy': float(m.group(2))}),
]


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object."""
    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')


def parse_log_file(log_file, reference_time=None):
    """
    Parse training log and extract events.
    
    Args:
        log_file: Path to the training log file
        reference_time: Optional reference datetime for calculating elapsed time
        
    Returns:
        List of event dictionaries
    """
    events = []
    first_timestamp = None
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            for pattern, event_type, extractor in EVENT_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    timestamp = parse_timestamp(match.group(1))
                    
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    
                    ref_time = reference_time or first_timestamp
                    elapsed_sec = (timestamp - ref_time).total_seconds()
                    
                    event = {
                        'timestamp': timestamp.isoformat(),
                        'elapsed_sec': elapsed_sec,
                        'elapsed_hours': elapsed_sec / 3600,
                        'event_type': event_type,
                        'line_num': line_num,
                    }
                    event.update(extractor(match))
                    events.append(event)
                    break
    
    return events, first_timestamp


def create_event_label(event):
    """Create a short label for the event."""
    etype = event['event_type']
    
    if etype == 'validation_bpb':
        return f"Val bpb={event['value']:.3f} (step {event['step']})"
    elif etype == 'validation_loss':
        return f"Val loss={event['value']:.3f} (step {event['step']})"
    elif etype == 'core_metric':
        return f"CORE={event['value']:.3f} (step {event['step']})"
    elif etype == 'inline_benchmark':
        return f"MMLU={event['mmlu']:.2f}, ARC={event['arc_easy']:.2f} (step {event['step']})"
    elif etype == 'benchmark_eval':
        return f"Eval: {event['benchmark']} acc={event['accuracy']:.3f}"
    elif etype == 'checkpoint_save':
        return f"Checkpoint saved"
    elif etype == 'model_load':
        return f"Model loaded (step {event['step']})"
    elif etype == 'arc_easy_result':
        return f"ARC-Easy: {event['accuracy']:.1f}%"
    elif etype == 'mmlu_result':
        return f"MMLU: {event['accuracy']:.1f}%"
    elif etype == 'humaneval_result':
        return f"HumanEval: {event['accuracy']:.1f}%"
    else:
        return etype


def save_events_csv(events, output_file):
    """Save events to CSV file."""
    if not events:
        print("No events found")
        return

    # Add label to each event
    for event in events:
        event['label'] = create_event_label(event)

    # Get all unique keys
    all_keys = set()
    for event in events:
        all_keys.update(event.keys())

    fieldnames = ['timestamp', 'elapsed_sec', 'elapsed_hours', 'event_type', 'label', 'line_num']
    fieldnames += sorted(k for k in all_keys if k not in fieldnames)

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(events)

    print(f"Saved {len(events)} events to {output_file}")


def print_summary(events):
    """Print summary of extracted events."""
    from collections import Counter
    counts = Counter(e['event_type'] for e in events)

    print("\nEvent Summary:")
    print("-" * 40)
    for event_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {event_type}: {count}")
    print("-" * 40)
    print(f"  Total: {len(events)} events")

    if events:
        duration_hours = events[-1]['elapsed_hours']
        print(f"  Duration: {duration_hours:.2f} hours")


def main():
    parser = argparse.ArgumentParser(
        description='Parse training log and extract significant events.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('log_file', type=str, help='Path to training log file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file (default: <log_file>_events.csv)')
    parser.add_argument('--reference-time', type=str, default=None,
                        help='Reference time for elapsed calculation (format: YYYY-MM-DD HH:MM:SS)')

    args = parser.parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return 1

    output_file = args.output or str(log_file.with_suffix('')) + '_events.csv'

    ref_time = None
    if args.reference_time:
        ref_time = parse_timestamp(args.reference_time)

    print(f"Parsing: {log_file}")
    events, first_ts = parse_log_file(log_file, reference_time=ref_time)

    if first_ts:
        print(f"First event timestamp: {first_ts}")

    print_summary(events)
    save_events_csv(events, output_file)

    return 0


if __name__ == '__main__':
    exit(main())

