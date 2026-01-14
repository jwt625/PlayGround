# Augment Conversation History Extraction

## Overview

This document describes how Augment stores conversation histories in VSCode and how to extract them for analysis or backup purposes.

## Storage Location and Structure

### Primary Storage Path
```
~/Library/Application Support/Code/User/workspaceStorage/
```

### Directory Organization

Each workspace (project folder) gets a unique workspace ID directory:
```
workspaceStorage/
├── <workspace-id-1>/
│   ├── workspace.json                    # Maps workspace ID to folder path
│   ├── Augment.vscode-augment/
│   │   ├── augment-kv-store/            # LevelDB database with conversations
│   │   │   ├── *.ldb                    # LevelDB table files (binary)
│   │   │   ├── *.log                    # LevelDB log files (binary)
│   │   │   ├── MANIFEST-*               # Database manifest
│   │   │   ├── CURRENT                  # Current manifest pointer
│   │   │   └── LOCK                     # Database lock file
│   │   ├── augment-global-state/        # UI state, file indices
│   │   ├── augment-user-assets/         # Tasks, checkpoints, untruncated content
│   │   └── Augment-Memories             # User memories
│   └── chatSessions/                     # Legacy chat session files (JSON)
├── <workspace-id-2>/
└── ...
```

### Workspace Identification

The `workspace.json` file maps workspace IDs to project folders:
```json
{
  "folder": "file:///Users/username/Documents/GitHub/ProjectName"
}
```

For remote workspaces:
```json
{
  "folder": "vscode-remote://ssh-remote+hostname/path/to/project"
}
```

## Data Storage Format

### LevelDB Key-Value Store

Augment uses LevelDB to store conversation data in `augment-kv-store/`. The database contains:

**Key Patterns:**
- `exchange:<conversation-id>:<exchange-id>` - Individual conversation exchanges
- `metadata:<conversation-id>` - Conversation metadata (exchange count, timestamps)
- `tooluse:<conversation-id>:<request-id>;<tool-use-id>` - Tool execution state

**Value Format:**
All values are JSON-encoded strings containing conversation data.

### Conversation Data Structure

**Exchange Object:**
```json
{
  "uuid": "exchange-uuid",
  "conversationId": "conversation-uuid",
  "request_message": "User's question or request",
  "response_text": "Assistant's response",
  "request_nodes": [...],
  "response_nodes": [...],
  "model_id": "claude-sonnet-4-5",
  "status": "success|sent|error",
  "timestamp": "2026-01-13T18:24:31.388Z"
}
```

**Metadata Object:**
```json
{
  "conversationId": "conversation-uuid",
  "totalExchanges": 5,
  "lastUpdated": 1768328671401
}
```

## Extraction Process

### Why Direct LevelDB Reading Fails

The Python `plyvel` library requires LevelDB C++ headers to be installed. On macOS, this often fails during compilation. Instead, we use a binary parsing approach.

### Extraction Method

The extraction script (`extract_augment_conversations.py`) works by:

1. **Scanning workspace directories** - Finds all VSCode workspaces with Augment data
2. **Reading binary files** - Opens `.ldb` and `.log` files in the kv-store
3. **Parsing JSON objects** - Extracts JSON objects from binary data using bracket matching
4. **Organizing by conversation** - Groups exchanges by conversation ID
5. **Exporting to JSON** - Saves organized data to readable JSON files

### Key Algorithm: JSON Extraction from Binary

```python
def extract_json_objects(text, min_length=50):
    """Extract JSON objects from text using bracket matching."""
    objects = []
    depth = 0
    start = None
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start is not None:
                json_str = text[start:i+1]
                if len(json_str) >= min_length:
                    try:
                        obj = json.loads(json_str)
                        objects.append(obj)
                    except:
                        pass
                start = None
    
    return objects
```

## Using the Extraction Script

### Location
```
20260113_augment_export/extract_augment_conversations.py
```

### Running the Script
```bash
cd 20260113_augment_export
python3 extract_augment_conversations.py
```

### Output Structure
```
augment_conversations_export/
├── extraction_summary.json                           # Overall summary
├── <workspace-id>_<folder-path>.json                # Per-workspace conversations
└── ...
```

### Output File Format
```json
{
  "workspace_id": "e415494a495aa0c96d6622ddc57c4a1b",
  "folder_path": "/Users/username/Documents/GitHub/ProjectName",
  "extracted_at": "2026-01-13T12:02:16.937049",
  "total_items": 50,
  "conversation_count": 1,
  "conversations": [
    {
      "conversation_id": "cea62c50-908f-4d6c-aefa-aa62365edc0d",
      "exchanges": [...],
      "metadata": [...]
    }
  ],
  "other_items": [...]
}
```

## Finding Example Raw Data

### Locate Current Workspace
```bash
# Find workspace ID for current project
cd ~/Library/Application\ Support/Code/User/workspaceStorage
grep -r "PlayGround" */workspace.json
```

### Inspect Raw LevelDB Files
```bash
# View raw data (will show binary + text)
strings <workspace-id>/Augment.vscode-augment/augment-kv-store/000100.log | head -100

# Search for specific conversation
strings <workspace-id>/Augment.vscode-augment/augment-kv-store/*.ldb | grep "conversationId"
```

### Example: Current Workspace
For the PlayGround project, the workspace ID is `e415494a495aa0c96d6622ddc57c4a1b`:
```bash
cd ~/Library/Application\ Support/Code/User/workspaceStorage/e415494a495aa0c96d6622ddc57c4a1b
ls -la Augment.vscode-augment/augment-kv-store/
```

## Modifying the Extraction Script

### Adding New Data Types

To extract additional data types, modify the filter in `extract_conversations_from_kv_store()`:

```python
# Current filter
if any(key in obj for key in ['conversationId', 'uuid', 'request_message', 
                                'response_text', 'request_nodes', 'response_nodes']):
    all_data.append(obj)

# Add new keys to extract
if any(key in obj for key in ['conversationId', 'uuid', 'request_message', 
                                'response_text', 'request_nodes', 'response_nodes',
                                'your_new_key']):  # Add here
    all_data.append(obj)
```

### Filtering by Date Range

Add timestamp filtering in the main loop:

```python
from datetime import datetime

# After extracting exchanges
for item in all_data:
    timestamp_str = item.get('timestamp')
    if timestamp_str:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        if start_date <= timestamp <= end_date:
            filtered_data.append(item)
```

### Exporting to Different Formats

Add export functions after the JSON export:

```python
# Export to CSV
import csv
with open(output_file.with_suffix('.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['conversation_id', 'timestamp', 'request_message', 'response_text'])
    writer.writeheader()
    for conv in conversations_by_id.values():
        for exchange in conv['exchanges']:
            writer.writerow({
                'conversation_id': conv['conversation_id'],
                'timestamp': exchange.get('timestamp', ''),
                'request_message': exchange.get('request_message', ''),
                'response_text': exchange.get('response_text', '')
            })
```

## Troubleshooting

### No Conversations Found

Check if Augment data exists:
```bash
ls ~/Library/Application\ Support/Code/User/workspaceStorage/*/Augment.vscode-augment/
```

### Incomplete Extraction

The LevelDB database may be locked if VSCode is running. Close VSCode before extraction:
```bash
# Check for locks
ls -la ~/Library/Application\ Support/Code/User/workspaceStorage/*/Augment.vscode-augment/augment-kv-store/LOCK
```

### Binary Parsing Issues

If JSON extraction fails, increase the minimum length threshold or adjust the bracket matching logic in `extract_json_objects()`.

## Project Organization by Folder

Augment organizes conversations by the workspace folder path. When you open a different folder in VSCode:

1. VSCode creates a new workspace ID (or reuses existing one)
2. Augment stores conversations in that workspace's kv-store
3. Only conversations from that workspace are visible in the UI

This means:
- Opening `/path/to/ProjectA` shows only ProjectA conversations
- Opening `/path/to/ProjectB` shows only ProjectB conversations
- Each project's history is isolated and stored separately

## Notes

- The extraction is read-only and does not modify the original database
- LevelDB files are binary and cannot be edited directly
- Conversation IDs are UUIDs that persist across sessions
- Exchange order is preserved by timestamp fields
- The script handles both local and remote workspace paths

## Bug Fix: Binary Parsing vs Proper LevelDB SDK

### The Problem

The original Python extraction script (`extract_augment_conversations.py`) used a naive binary parsing approach:
1. Read `.ldb` and `.log` files as raw bytes
2. Decode as UTF-8 with error handling
3. Use bracket matching to find JSON objects

This approach had a critical flaw: **LevelDB uses Snappy compression** for many of its data blocks. When compression is enabled, JSON data is fragmented across the binary with compression markers interspersed, making bracket matching fail.

### Symptoms

- Many workspaces reported "No conversations found" despite having data
- Successfully extracted workspaces had far fewer conversations than visible in the Augment UI
- Hexdump analysis showed JSON fragments with binary markers like `82 35 00 4c` breaking up the data

### The Fix

Replaced the binary parsing approach with a proper LevelDB SDK:

```bash
# Install Node.js LevelDB library
npm install classic-level
```

New extraction script (`extract_with_leveldb.js`) uses the `classic-level` package which:
- Properly handles LevelDB's internal format
- Automatically decompresses Snappy-compressed blocks
- Correctly iterates through all key-value pairs

### Results Comparison

| Metric | Binary Parsing | Proper LevelDB | Improvement |
|--------|----------------|----------------|-------------|
| Workspaces extracted | 14 | 21 | +50% |
| Total exchanges | 414 | 52,254 | **+126x** |

The proper LevelDB extraction recovered **126 times more data** than the binary parsing approach.

### Known Limitations

Workspaces currently open in VSCode may have locked databases, causing iterator errors. Close VSCode before extraction to access all data.

### Lesson Learned

When dealing with database formats, always use the proper SDK/library rather than attempting to parse binary files directly. Database formats often include compression, checksums, and internal structures that are not designed for direct reading.
