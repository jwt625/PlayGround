#!/usr/bin/env python3
"""
Extract Augment conversation histories from VSCode workspace storage.

This script reads the LevelDB database files where Augment stores conversation data
and extracts them into readable JSON files organized by workspace/project.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

# VSCode workspace storage path
VSCODE_STORAGE = Path.home() / "Library/Application Support/Code/User/workspaceStorage"

def find_workspace_folders():
    """Find all workspace folders with their associated project paths."""
    workspaces = {}
    
    for workspace_dir in VSCODE_STORAGE.iterdir():
        if not workspace_dir.is_dir():
            continue
            
        workspace_json = workspace_dir / "workspace.json"
        if workspace_json.exists():
            try:
                with open(workspace_json, 'r') as f:
                    data = json.load(f)
                    folder = data.get('folder', '')
                    # Extract folder path from URI
                    if folder.startswith('file://'):
                        folder_path = folder.replace('file://', '')
                    elif folder.startswith('vscode-remote://'):
                        folder_path = folder
                    else:
                        folder_path = folder
                    
                    workspaces[workspace_dir.name] = {
                        'path': folder_path,
                        'workspace_dir': workspace_dir
                    }
            except Exception as e:
                print(f"Error reading {workspace_json}: {e}")
    
    return workspaces

def extract_json_objects(text, min_length=50):
    """Extract JSON objects from text using a more robust approach."""
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

def extract_conversations_from_kv_store(kv_store_path):
    """Extract conversation data from LevelDB kv-store by reading raw files."""
    all_data = []

    if not kv_store_path.exists():
        return all_data

    # Read all .ldb and .log files
    for file in list(kv_store_path.glob("*.ldb")) + list(kv_store_path.glob("*.log")):
        try:
            with open(file, 'rb') as f:
                content = f.read()
                # Decode with error handling
                text = content.decode('utf-8', errors='ignore')

                # Extract all JSON objects
                objects = extract_json_objects(text)

                for obj in objects:
                    # Filter for conversation-related objects
                    if any(key in obj for key in ['conversationId', 'uuid', 'request_message',
                                                    'response_text', 'request_nodes', 'response_nodes']):
                        all_data.append(obj)

        except Exception as e:
            print(f"  Warning: Error reading {file.name}: {e}")

    return all_data

def main():
    print("Extracting Augment conversation histories...")
    print(f"VSCode storage path: {VSCODE_STORAGE}")
    print()
    
    workspaces = find_workspace_folders()
    print(f"Found {len(workspaces)} workspaces")
    print()
    
    output_dir = Path("augment_conversations_export")
    output_dir.mkdir(exist_ok=True)
    
    summary = []
    
    for workspace_id, info in workspaces.items():
        folder_path = info['path']
        workspace_dir = info['workspace_dir']
        
        # Check if Augment data exists
        augment_dir = workspace_dir / "Augment.vscode-augment"
        if not augment_dir.exists():
            continue
        
        kv_store = augment_dir / "augment-kv-store"
        
        print(f"Processing: {folder_path}")
        print(f"  Workspace ID: {workspace_id}")
        
        all_data = extract_conversations_from_kv_store(kv_store)

        if all_data:
            # Organize data by conversation ID
            conversations_by_id = {}
            metadata_items = []

            for item in all_data:
                conv_id = item.get('conversationId')
                if conv_id:
                    if conv_id not in conversations_by_id:
                        conversations_by_id[conv_id] = {
                            'conversation_id': conv_id,
                            'exchanges': [],
                            'metadata': []
                        }

                    # Check if this is an exchange or metadata
                    if 'request_message' in item or 'response_text' in item:
                        conversations_by_id[conv_id]['exchanges'].append(item)
                    else:
                        conversations_by_id[conv_id]['metadata'].append(item)
                else:
                    metadata_items.append(item)

            # Create output file
            safe_name = folder_path.replace('/', '_').replace(':', '_')
            output_file = output_dir / f"{workspace_id}_{safe_name}.json"

            with open(output_file, 'w') as f:
                json.dump({
                    'workspace_id': workspace_id,
                    'folder_path': folder_path,
                    'extracted_at': datetime.now().isoformat(),
                    'total_items': len(all_data),
                    'conversation_count': len(conversations_by_id),
                    'conversations': list(conversations_by_id.values()),
                    'other_items': metadata_items
                }, f, indent=2)

            total_exchanges = sum(len(c['exchanges']) for c in conversations_by_id.values())
            print(f"  Extracted {len(conversations_by_id)} conversations with {total_exchanges} exchanges")
            print(f"  Saved to: {output_file}")

            summary.append({
                'workspace_id': workspace_id,
                'folder_path': folder_path,
                'conversation_count': len(conversations_by_id),
                'exchange_count': total_exchanges,
                'output_file': str(output_file)
            })
        else:
            print(f"  No conversations found")
        
        print()
    
    # Save summary
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'extracted_at': datetime.now().isoformat(),
            'total_workspaces': len(summary),
            'workspaces': summary
        }, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"Total workspaces with conversations: {len(summary)}")

if __name__ == "__main__":
    main()

