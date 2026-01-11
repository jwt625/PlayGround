#!/usr/bin/env python3
"""
Extract ALL entities from Claude Code workflow logs.

This script identifies and extracts all entities created during the workflow:
1. API Requests - Each HTTP request to the inference endpoint
2. API Responses - Each response from the inference endpoint
3. Conversations - Multi-turn conversations (message history)
4. Messages - Individual messages (user/assistant)
5. Tool Definitions - Available tools in each request
6. Tool Uses - When the assistant invokes a tool
7. Tool Results - Results returned from tool execution
8. Tasks - Subagent tasks (special case of Tool Use with name="Task")
9. Agents - Inferred from Task tool uses (main agent + subagents)
10. System Prompts - Instructions given to agents
11. Content Blocks - Text, tool_use, tool_result blocks in messages
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from datetime import datetime
import hashlib

# Import agent tracking and deduplication
from .agent_tracker import AgentInstanceTracker
from .entity_deduplicator import EntityDeduplicator


def compute_hash(text: str, length: int = 16) -> str:
    """Compute a short hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


class EntityExtractor:
    """Extract all entities from Claude Code workflow logs."""

    def __init__(self):
        # Entity storage
        self.api_requests = []
        self.api_responses = []
        self.conversations = []
        self.messages = []
        self.tool_definitions = {}  # tool_name -> definition
        self.tool_uses = []
        self.tool_results = []
        self.tasks = []  # Special: Task tool uses
        self.agents = {}  # agent_id -> agent info
        self.system_prompts = {}  # hash -> prompt text
        self.content_blocks = []

        # Relationship tracking
        self.request_to_response = {}  # request_idx -> response_idx
        self.tool_use_to_result = {}  # tool_use_id -> tool_result
        self.task_to_agent = {}  # task_tool_use_id -> agent_id
        self.conversation_messages = defaultdict(list)  # conv_id -> [message_ids]

        # Counters
        self.request_counter = 0
        self.message_counter = 0
        self.content_block_counter = 0

        # Agent tracking and deduplication
        self.agent_tracker = AgentInstanceTracker()
        self.deduplicator = EntityDeduplicator(self.agent_tracker)
        
    def extract_from_log_file(self, log_path: Path):
        """Extract all entities from a single log file."""
        print(f"Processing {log_path}...")
        
        with open(log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    self.process_log_entry(entry, line_num)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}", file=sys.stderr)
    
    def process_log_entry(self, entry: Dict[str, Any], line_num: int):
        """Process a single log entry and extract all entities."""
        timestamp = entry.get('timestamp', '')
        body = entry.get('body', {})

        # Extract API Request
        request_id = self.request_counter
        self.request_counter += 1

        # IDENTIFY AGENT INSTANCE (with timestamp)
        agent_instance = self.agent_tracker.identify_or_create_agent(request_id, body, timestamp)

        request_entity = {
            'id': f'req_{request_id}',
            'line_num': line_num,
            'timestamp': timestamp,
            'method': entry.get('method'),
            'path': entry.get('path'),
            'url': entry.get('url'),
            'headers': entry.get('headers', {}),
            'body': body,
            # Agent tracking metadata
            'agent_id': agent_instance.agent_id,
            'agent_type': agent_instance.system_prompt_hash,
            'is_continuation': len(agent_instance.requests) > 1,
            'conversation_turn': len(agent_instance.message_count_history),
            'spawned_by_task': agent_instance.spawned_by_task_id,
            'parent_agent': agent_instance.parent_agent_id,
        }
        self.api_requests.append(request_entity)

        # Extract System Prompt
        if 'system' in body:
            self.extract_system_prompt(body['system'], request_id)

        # Extract Tool Definitions
        if 'tools' in body:
            self.extract_tool_definitions(body['tools'], request_id)

        # Extract Messages (conversation history)
        if 'messages' in body:
            self.extract_messages(body['messages'], request_id, timestamp)
            # Track request content for content reuse detection
            self.agent_tracker.track_request_content(request_id, body['messages'], timestamp)

        # Extract API Response
        if 'response' in entry:
            response_entity = self.extract_response(entry['response'], request_id, timestamp)
            self.request_to_response[request_id] = len(self.api_responses) - 1

            # Track response content for content reuse detection
            response_body = entry['response'].get('body', {})
            if 'content' in response_body:
                self.agent_tracker.track_response_content(request_id, response_body['content'], timestamp)
    
    def extract_system_prompt(self, system: List[Dict], request_id: int):
        """Extract system prompt from request."""
        texts = []
        for prompt in system:
            if isinstance(prompt, dict):
                text = prompt.get('text', '')
                if text:
                    texts.append(text)
        
        if texts:
            combined = '|||'.join(texts)
            prompt_hash = compute_hash(combined, length=16)
            if prompt_hash not in self.system_prompts:
                self.system_prompts[prompt_hash] = {
                    'hash': prompt_hash,
                    'text': combined,
                    'first_seen_request': request_id,
                }
    
    def extract_tool_definitions(self, tools: List[Dict], request_id: int):
        """Extract tool definitions from request."""
        for tool in tools:
            if isinstance(tool, dict):
                tool_name = tool.get('name')
                if tool_name:
                    if tool_name not in self.tool_definitions:
                        self.tool_definitions[tool_name] = {
                            'name': tool_name,
                            'description': tool.get('description', ''),
                            'input_schema': tool.get('input_schema', {}),
                            'first_seen_request': request_id,
                        }

    def extract_messages(self, messages: List[Dict], request_id: int, timestamp: str):
        """Extract messages from conversation history."""
        for msg_idx, msg in enumerate(messages):
            message_id = f'msg_{self.message_counter}'
            self.message_counter += 1

            role = msg.get('role')
            content = msg.get('content')

            message_entity = {
                'id': message_id,
                'request_id': request_id,
                'role': role,
                'timestamp': timestamp,
                'position_in_conversation': msg_idx,
                'content_type': type(content).__name__,
                'content_blocks': [],
            }

            # Extract content blocks
            if isinstance(content, str):
                # Simple text message
                block_id = f'block_{self.content_block_counter}'
                self.content_block_counter += 1
                block = {
                    'id': block_id,
                    'message_id': message_id,
                    'type': 'text',
                    'text': content,
                }
                self.content_blocks.append(block)
                message_entity['content_blocks'].append(block_id)

            elif isinstance(content, list):
                # Structured content with multiple blocks
                for block_idx, item in enumerate(content):
                    if isinstance(item, dict):
                        block_id = self.extract_content_block(item, message_id, block_idx, request_id)
                        message_entity['content_blocks'].append(block_id)

            self.messages.append(message_entity)

    def extract_content_block(self, block: Dict, message_id: str, position: int, request_id: int = None) -> str:
        """Extract a single content block (text, tool_use, tool_result)."""
        block_id = f'block_{self.content_block_counter}'
        self.content_block_counter += 1

        block_type = block.get('type')

        block_entity = {
            'id': block_id,
            'message_id': message_id,
            'position': position,
            'type': block_type,
        }

        if block_type == 'text':
            block_entity['text'] = block.get('text', '')

        elif block_type == 'tool_use':
            tool_use_id = block.get('id')
            tool_name = block.get('name')
            tool_input = block.get('input', {})

            block_entity['tool_use_id'] = tool_use_id
            block_entity['tool_name'] = tool_name
            block_entity['tool_input'] = tool_input

            # Create tool use entity
            tool_use_entity = {
                'id': tool_use_id,
                'block_id': block_id,
                'message_id': message_id,
                'tool_name': tool_name,
                'input': tool_input,
            }

            # Apply deduplication if request_id available
            if request_id is not None:
                tool_use_entity = self.deduplicator.deduplicate_entity(
                    tool_use_entity, 'tool_use', request_id
                )

                # Track tool use in agent tracker (for workflow DAG)
                # Get timestamp from request
                timestamp = ""
                if request_id < len(self.api_requests):
                    timestamp = self.api_requests[request_id].get('timestamp', '')
                self.agent_tracker.track_tool_use(request_id, block, timestamp)

            self.tool_uses.append(tool_use_entity)

            # Special handling for Task tool
            if tool_name == 'Task':
                task_prompt = tool_input.get('prompt', '')
                task_entity = {
                    'id': tool_use_id,
                    'tool_use_id': tool_use_id,
                    'description': tool_input.get('description', ''),
                    'prompt': task_prompt,
                    'subagent_type': tool_input.get('subagent_type', ''),
                }

                # Apply deduplication
                if request_id is not None:
                    task_entity = self.deduplicator.deduplicate_entity(
                        task_entity, 'task', request_id
                    )

                    # Register task prompt for subagent matching (only for first occurrence)
                    if not task_entity.get('is_duplicate'):
                        agent_id = self.agent_tracker.request_to_agent.get(request_id)
                        if agent_id and task_prompt:
                            self.agent_tracker.register_task_prompt(tool_use_id, task_prompt, agent_id)

                self.tasks.append(task_entity)

        elif block_type == 'tool_result':
            tool_use_id = block.get('tool_use_id')
            result_content = block.get('content')

            block_entity['tool_use_id'] = tool_use_id
            block_entity['result_content'] = result_content

            # Create tool result entity
            tool_result_entity = {
                'id': f'result_{len(self.tool_results)}',
                'block_id': block_id,
                'message_id': message_id,
                'tool_use_id': tool_use_id,
                'content': result_content,
            }
            self.tool_results.append(tool_result_entity)
            self.tool_use_to_result[tool_use_id] = len(self.tool_results) - 1

            # Track tool result in agent tracker (for workflow DAG)
            if request_id is not None:
                timestamp = ""
                if request_id < len(self.api_requests):
                    timestamp = self.api_requests[request_id].get('timestamp', '')
                self.agent_tracker.track_tool_result(request_id, block, timestamp)

            # Extract agent ID from Task tool results
            if isinstance(result_content, list):
                for item in result_content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        if 'agentId:' in text:
                            # Extract agent ID
                            import re
                            match = re.search(r'agentId:\s*([a-f0-9]+)', text)
                            if match:
                                agent_id = match.group(1)
                                self.task_to_agent[tool_use_id] = agent_id
                                if agent_id not in self.agents:
                                    self.agents[agent_id] = {
                                        'id': agent_id,
                                        'task_tool_use_id': tool_use_id,
                                        'first_seen': message_id,
                                    }

        self.content_blocks.append(block_entity)
        return block_id

    def extract_response(self, response: Dict, request_id: int, timestamp: str) -> Dict:
        """Extract response entity and its content."""
        response_entity = {
            'id': f'resp_{len(self.api_responses)}',
            'request_id': request_id,
            'timestamp': response.get('timestamp', timestamp),
            'status': response.get('status'),
            'duration_ms': response.get('duration_ms'),
            'body': response.get('body', {}),
        }

        # Extract response content blocks (assistant's response)
        body = response.get('body', {})
        if 'content' in body:
            # Create a synthetic message for the assistant's response
            message_id = f'msg_{self.message_counter}'
            self.message_counter += 1

            message_entity = {
                'id': message_id,
                'request_id': request_id,
                'role': 'assistant',
                'timestamp': timestamp,
                'position_in_conversation': -1,  # Response, not in request messages
                'content_type': 'list',
                'content_blocks': [],
                'response_id': response_entity['id'],
            }

            for block_idx, item in enumerate(body['content']):
                if isinstance(item, dict):
                    block_id = self.extract_content_block(item, message_id, block_idx, request_id)
                    message_entity['content_blocks'].append(block_id)

            self.messages.append(message_entity)

        self.api_responses.append(response_entity)
        return response_entity

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of all extracted entities."""
        agent_stats = self.agent_tracker.get_statistics()
        dedup_stats = self.deduplicator.get_deduplication_stats()

        return {
            'counts': {
                'api_requests': len(self.api_requests),
                'api_responses': len(self.api_responses),
                'messages': len(self.messages),
                'content_blocks': len(self.content_blocks),
                'tool_definitions': len(self.tool_definitions),
                'tool_uses': len(self.tool_uses),
                'tool_results': len(self.tool_results),
                'tasks': len(self.tasks),
                'agents': len(self.agents),
                'system_prompts': len(self.system_prompts),
            },
            'agent_tracking': agent_stats,
            'deduplication': dedup_stats,
            'tool_usage': self.get_tool_usage_stats(),
            'task_types': self.get_task_type_stats(),
            'agent_info': list(self.agents.values()),
        }

    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get statistics on tool usage."""
        tool_counts = defaultdict(int)
        for tool_use in self.tool_uses:
            tool_name = tool_use.get('tool_name') or 'unknown'
            tool_counts[tool_name] += 1
        return dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True))

    def get_task_type_stats(self) -> Dict[str, int]:
        """Get statistics on task subagent types."""
        type_counts = defaultdict(int)
        for task in self.tasks:
            subagent_type = task.get('subagent_type', 'unknown')
            type_counts[subagent_type] += 1
        return dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))

    def export_to_json(self, output_path: Path):
        """Export all entities to JSON file."""
        agent_hierarchy = self.agent_tracker.get_agent_hierarchy()
        workflow_dag = self.agent_tracker.build_workflow_dag()

        data = {
            'metadata': {
                'extraction_timestamp': datetime.now().isoformat(),
                'summary': self.generate_summary(),
            },
            'entities': {
                'api_requests': self.api_requests,
                'api_responses': self.api_responses,
                'messages': self.messages,
                'content_blocks': self.content_blocks,
                'tool_definitions': list(self.tool_definitions.values()),
                'tool_uses': self.tool_uses,
                'tool_results': self.tool_results,
                'tasks': self.tasks,
                'agents': list(self.agents.values()),
                'system_prompts': list(self.system_prompts.values()),
                'agent_instances': self.agent_tracker.export_all_instances(),
            },
            'relationships': {
                'request_to_response': self.request_to_response,
                'tool_use_to_result': self.tool_use_to_result,
                'task_to_agent': self.task_to_agent,
                'agent_hierarchy': agent_hierarchy,
                'request_to_agent': self.agent_tracker.request_to_agent,
            },
            'workflow_dag': workflow_dag,
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nExported all entities to {output_path}")

    def print_summary(self):
        """Print summary of extracted entities."""
        summary = self.generate_summary()

        print("\n" + "="*80)
        print("ENTITY EXTRACTION SUMMARY")
        print("="*80)

        print("\nEntity Counts:")
        for entity_type, count in summary['counts'].items():
            print(f"  {entity_type:20s}: {count:5d}")

        print("\n" + "="*80)
        print("AGENT TRACKING")
        print("="*80)
        agent_stats = summary['agent_tracking']
        print(f"  Total Agent Instances: {agent_stats['total_agents']}")
        print(f"  Total Requests: {agent_stats['total_requests']}")
        print(f"  Avg Requests/Agent: {agent_stats['avg_requests_per_agent']:.2f}")
        print(f"  Root Agents: {agent_stats['root_agents']}")
        print(f"  Child Agents (spawned): {agent_stats['child_agents']}")
        print(f"  Unique Agent Types: {agent_stats['unique_agent_types']}")

        print("\n" + "="*80)
        print("DEDUPLICATION STATISTICS")
        print("="*80)
        dedup_stats = summary['deduplication']
        print(f"  Total Unique Entities: {dedup_stats['total_unique_entities']}")
        print(f"  Total Occurrences: {dedup_stats['total_occurrences']}")
        print(f"  Overall Duplication Ratio: {dedup_stats['overall_duplication_ratio']:.2f}x")
        print(f"  Duplicates Removed: {dedup_stats['duplicates_removed']}")

        print("\n  By Entity Type:")
        for entity_type, stats in dedup_stats['by_entity_type'].items():
            print(f"    {entity_type:15s}: {stats['unique']:4d} unique, {stats['total']:5d} total ({stats['duplication_ratio']:.1f}x)")

        print("\n" + "="*80)
        print("TOOL USAGE (top 10)")
        print("="*80)
        for tool_name, count in list(summary['tool_usage'].items())[:10]:
            print(f"  {tool_name:30s}: {count:5d}")

        print("\nTask Subagent Types:")
        for subagent_type, count in summary['task_types'].items():
            print(f"  {subagent_type:30s}: {count:5d}")

        print("\nSystem Prompts:")
        for prompt_hash, prompt_info in self.system_prompts.items():
            text_preview = prompt_info['text'][:100].replace('\n', ' ')
            print(f"  {prompt_hash}: {text_preview}...")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract all entities from Claude Code workflow logs'
    )
    parser.add_argument(
        'log_file',
        type=Path,
        help='Path to JSONL log file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output JSON file (default: entities_extracted.json)'
    )

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.log_file.parent / 'entities_extracted.json'

    # Extract entities
    extractor = EntityExtractor()
    extractor.extract_from_log_file(args.log_file)

    # Print summary
    extractor.print_summary()

    # Export to JSON
    extractor.export_to_json(output_path)

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

