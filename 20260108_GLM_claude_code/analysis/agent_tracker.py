#!/usr/bin/env python3
"""
Agent Instance Tracking for Claude Code workflow logs.

Identifies and tracks agent instances across multiple API requests by analyzing
conversation history patterns and fingerprinting conversation states.
"""

import hashlib
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


def compute_hash(text: str, length: int = 16) -> str:
    """Compute a short hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


@dataclass
class AgentInstance:
    """Represents a single agent instance across multiple API requests."""
    agent_id: str
    system_prompt_hash: str
    conversation_fingerprint: str
    requests: List[int] = field(default_factory=list)
    first_request_id: int = None
    last_request_id: int = None
    message_count_history: List[int] = field(default_factory=list)
    spawned_by_task_id: Optional[str] = None
    parent_agent_id: Optional[str] = None
    first_user_message: str = ""

    # Workflow tracking - NEW
    child_agent_ids: List[str] = field(default_factory=list)  # Agents spawned by this agent
    tool_uses: List[Dict[str, Any]] = field(default_factory=list)  # Tool uses by this agent
    tool_results: List[Dict[str, Any]] = field(default_factory=list)  # Tool results received
    timestamps: List[str] = field(default_factory=list)  # Timestamp for each request

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'system_prompt_hash': self.system_prompt_hash,
            'conversation_fingerprint': self.conversation_fingerprint,
            'requests': self.requests,
            'first_request_id': self.first_request_id,
            'last_request_id': self.last_request_id,
            'message_count_history': self.message_count_history,
            'timestamps': self.timestamps,  # Include full timestamps list
            'spawned_by_task_id': self.spawned_by_task_id,
            'parent_agent_id': self.parent_agent_id,
            'child_agent_ids': self.child_agent_ids,
            'first_user_message': self.first_user_message[:200],  # Truncate for readability
            'total_requests': len(self.requests),
            'conversation_turns': len(self.message_count_history),
            'tool_use_count': len(self.tool_uses),
            'tool_result_count': len(self.tool_results),
            'first_timestamp': self.timestamps[0] if self.timestamps else None,
            'last_timestamp': self.timestamps[-1] if self.timestamps else None,
        }


def compute_conversation_fingerprint(messages: List[Dict]) -> str:
    """
    Compute a unique fingerprint for a conversation based on message sequence.

    This creates a hash representing the conversation state based on CONTENT only,
    not unique IDs or formatting. This allows identifying the same agent across
    different tool invocations and content format variations.

    Args:
        messages: List of message dictionaries from API request

    Returns:
        16-character hex hash representing the conversation state
    """
    if not messages:
        return compute_hash("empty", length=16)

    fingerprint_parts = []

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', [])

        # Normalize content to list of blocks
        if isinstance(content, str):
            # Convert string content to structured format
            content = [{'type': 'text', 'text': content}]
        elif not isinstance(content, list):
            content = []

        # Process structured content - use role + block types + content hashes (NOT IDs)
        block_signature = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get('type', 'unknown')

                if block_type == 'text':
                    # For text blocks, use truncated hash
                    text = block.get('text', '')
                    text_hash = compute_hash(text, length=8)
                    block_signature.append(f"text:{text_hash}")

                elif block_type == 'tool_use':
                    # For tool_use, use tool_name + input hash (NOT tool_use_id)
                    tool_name = block.get('name', '')
                    tool_input = block.get('input', {})
                    input_hash = compute_hash(str(tool_input), length=8)
                    block_signature.append(f"tool_use:{tool_name}:{input_hash}")

                elif block_type == 'tool_result':
                    # For tool_result, use content hash (NOT tool_use_id reference)
                    result_content = block.get('content', '')
                    result_hash = compute_hash(str(result_content), length=8)
                    block_signature.append(f"tool_result:{result_hash}")

        fingerprint_parts.append(f"{role}:[{','.join(block_signature)}]")

    # Combine all message signatures and hash
    combined = '|||'.join(fingerprint_parts)
    return compute_hash(combined, length=16)


def extract_first_user_message(messages: List[Dict]) -> str:
    """Extract the first user message text from conversation."""
    if not messages:
        return ""
    
    first_msg = messages[0]
    if first_msg.get('role') != 'user':
        return ""
    
    content = first_msg.get('content', '')
    
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from first text block
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                return block.get('text', '')
    
    return ""


class AgentInstanceTracker:
    """Track agent instances across multiple API requests."""

    def __init__(self):
        self.instances: Dict[str, AgentInstance] = {}  # agent_id -> instance
        self.fingerprint_to_agent: Dict[str, str] = {}  # fingerprint -> agent_id
        self.request_to_agent: Dict[int, str] = {}  # request_id -> agent_id
        self.task_prompts: Dict[str, Dict[str, str]] = {}  # prompt_hash -> {task_id, agent_id}
        self.agent_counter = 0

        # Workflow tracking - NEW
        self.tool_use_index: Dict[str, Dict[str, Any]] = {}  # tool_use_id -> {agent_id, request_id, tool_name, timestamp}
        self.tool_result_index: Dict[str, Dict[str, Any]] = {}  # tool_use_id -> {agent_id, request_id, is_error, timestamp}
        self.workflow_edges: List[Dict[str, Any]] = []  # All edges in the workflow DAG

        # Content reuse tracking
        self.response_content_index: Dict[str, List[Dict[str, Any]]] = {}  # content_hash -> [{agent_id, request_id, timestamp}]

    def compute_system_prompt_hash(self, system: List[Dict]) -> str:
        """Compute hash of system prompt."""
        texts = []
        for prompt in system:
            if isinstance(prompt, dict):
                text = prompt.get('text', '')
                if text:
                    texts.append(text)

        if texts:
            combined = '|||'.join(texts)
            return compute_hash(combined, length=16)
        return 'no_system'

    def identify_or_create_agent(self, request_id: int, body: Dict, timestamp: str = "") -> AgentInstance:
        """
        Identify which agent instance this request belongs to.

        Strategy:
        1. Compute conversation fingerprint from messages
        2. Check if we've seen this exact conversation state before (replay)
        3. Check if this is a child of an existing conversation (grew by 1+ turns)
        4. Check if first message matches a Task tool prompt (subagent spawn)
        5. Otherwise, create a new agent instance

        Args:
            request_id: Unique request identifier
            body: Request body containing messages and system prompt

        Returns:
            AgentInstance object (existing or newly created)
        """
        messages = body.get('messages', [])
        system_prompt = body.get('system', [])

        # Compute fingerprints
        system_prompt_hash = self.compute_system_prompt_hash(system_prompt)
        conversation_fingerprint = compute_conversation_fingerprint(messages)

        # Check for exact match (same conversation state - replay)
        if conversation_fingerprint in self.fingerprint_to_agent:
            agent_id = self.fingerprint_to_agent[conversation_fingerprint]
            agent = self.instances[agent_id]
            agent.requests.append(request_id)
            agent.last_request_id = request_id
            agent.message_count_history.append(len(messages))
            agent.timestamps.append(timestamp)
            self.request_to_agent[request_id] = agent_id
            return agent

        # Check if this is a continuation (conversation grew)
        parent_agent = self.find_parent_conversation(messages, system_prompt_hash)

        if parent_agent:
            # This is the same agent, conversation just grew
            agent_id = parent_agent.agent_id
            agent = parent_agent
            agent.requests.append(request_id)
            agent.last_request_id = request_id
            agent.conversation_fingerprint = conversation_fingerprint
            agent.message_count_history.append(len(messages))
            agent.timestamps.append(timestamp)
            self.fingerprint_to_agent[conversation_fingerprint] = agent_id
            self.request_to_agent[request_id] = agent_id
            return agent

        # New agent instance
        agent_id = f"agent_{self.agent_counter}"
        self.agent_counter += 1

        first_user_msg = extract_first_user_message(messages)

        agent = AgentInstance(
            agent_id=agent_id,
            system_prompt_hash=system_prompt_hash,
            conversation_fingerprint=conversation_fingerprint,
            requests=[request_id],
            first_request_id=request_id,
            last_request_id=request_id,
            message_count_history=[len(messages)],
            first_user_message=first_user_msg,
            timestamps=[timestamp],
        )

        # Check if spawned by Task tool
        spawning_info = self.detect_task_spawn(first_user_msg)
        if spawning_info:
            agent.spawned_by_task_id = spawning_info['task_id']
            agent.parent_agent_id = spawning_info['parent_agent_id']

            # Add to parent's child list
            if agent.parent_agent_id and agent.parent_agent_id in self.instances:
                parent = self.instances[agent.parent_agent_id]
                if agent_id not in parent.child_agent_ids:
                    parent.child_agent_ids.append(agent_id)

        self.instances[agent_id] = agent
        self.fingerprint_to_agent[conversation_fingerprint] = agent_id
        self.request_to_agent[request_id] = agent_id

        return agent

    def find_parent_conversation(self, messages: List[Dict], system_prompt_hash: str) -> Optional[AgentInstance]:
        """
        Find if this conversation is a continuation of an existing agent.

        Strategy: Check if the first N-k messages match an existing agent's conversation,
        trying multiple backtrack depths (k=1,2,3,...) to handle tool use/result pairs.

        Args:
            messages: Current message list
            system_prompt_hash: System prompt hash for this request

        Returns:
            AgentInstance if parent found, None otherwise
        """
        if len(messages) <= 1:
            return None

        # Try backtracking by 1, 2, 3, ... messages to find parent conversation
        # This handles cases where conversation grew by multiple turns due to tool use/result
        max_backtrack = min(len(messages) - 1, 5)  # Try up to 5 messages back

        for backtrack in range(1, max_backtrack + 1):
            parent_fingerprint = compute_conversation_fingerprint(messages[:-backtrack])

            if parent_fingerprint in self.fingerprint_to_agent:
                agent_id = self.fingerprint_to_agent[parent_fingerprint]
                agent = self.instances[agent_id]

                # Verify system prompt matches (same agent type)
                if agent.system_prompt_hash == system_prompt_hash:
                    return agent

        return None

    def detect_task_spawn(self, first_user_message: str) -> Optional[Dict]:
        """
        Detect if this agent was spawned by a Task tool.

        Strategy: Check if the first user message matches a Task tool's prompt field.

        Args:
            first_user_message: First user message text

        Returns:
            Dict with task_id and parent_agent_id if match found, None otherwise
        """
        if not first_user_message:
            return None

        # Hash the first message
        msg_hash = compute_hash(first_user_message, length=16)

        # Look up in task prompts index
        if msg_hash in self.task_prompts:
            return self.task_prompts[msg_hash]

        return None

    def register_task_prompt(self, task_id: str, prompt: str, agent_id: str):
        """
        Register a Task tool prompt for later matching with subagent spawns.

        Args:
            task_id: Unique task identifier
            prompt: Task prompt text
            agent_id: Agent that created this task
        """
        if prompt:
            prompt_hash = compute_hash(prompt, length=16)
            self.task_prompts[prompt_hash] = {
                'task_id': task_id,
                'parent_agent_id': agent_id,
            }

    def get_agent_hierarchy(self) -> Dict[str, List[str]]:
        """
        Build parent-child agent hierarchy.

        Returns:
            Dict mapping parent_agent_id -> list of child_agent_ids
        """
        hierarchy = defaultdict(list)

        for agent_id, agent in self.instances.items():
            if agent.parent_agent_id:
                hierarchy[agent.parent_agent_id].append(agent_id)

        return dict(hierarchy)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked agents."""
        total_agents = len(self.instances)
        total_requests = sum(len(agent.requests) for agent in self.instances.values())

        root_agents = [a for a in self.instances.values() if not a.parent_agent_id]
        child_agents = [a for a in self.instances.values() if a.parent_agent_id]

        agent_types = defaultdict(int)
        for agent in self.instances.values():
            agent_types[agent.system_prompt_hash] += 1

        return {
            'total_agents': total_agents,
            'total_requests': total_requests,
            'avg_requests_per_agent': total_requests / total_agents if total_agents > 0 else 0,
            'root_agents': len(root_agents),
            'child_agents': len(child_agents),
            'unique_agent_types': len(agent_types),
            'agent_type_distribution': dict(agent_types),
        }

    def export_all_instances(self) -> List[Dict[str, Any]]:
        """Export all agent instances as dictionaries."""
        return [agent.to_dict() for agent in self.instances.values()]

    def track_tool_use(self, request_id: int, tool_use_block: Dict[str, Any], timestamp: str = ""):
        """
        Track a tool_use block from response content.

        Args:
            request_id: Request ID where tool was used
            tool_use_block: Tool use content block
            timestamp: Timestamp of the request
        """
        tool_use_id = tool_use_block.get('id')
        tool_name = tool_use_block.get('name')

        if not tool_use_id:
            return

        agent_id = self.request_to_agent.get(request_id)
        if not agent_id:
            return

        # Store in index
        self.tool_use_index[tool_use_id] = {
            'tool_use_id': tool_use_id,
            'tool_name': tool_name,
            'agent_id': agent_id,
            'request_id': request_id,
            'timestamp': timestamp,
            'input': tool_use_block.get('input', {}),
        }

        # Add to agent's tool_uses
        agent = self.instances[agent_id]
        agent.tool_uses.append({
            'tool_use_id': tool_use_id,
            'tool_name': tool_name,
            'request_id': request_id,
            'timestamp': timestamp,
        })

        # Special handling for Task tool (subagent spawn)
        if tool_name == 'Task':
            task_input = tool_use_block.get('input', {})
            prompt = task_input.get('prompt', '')
            subagent_type = task_input.get('subagent_type', '')

            if prompt:
                # Store for later matching with spawned agent
                prompt_hash = compute_hash(prompt, length=16)
                self.task_prompts[prompt_hash] = {
                    'task_id': tool_use_id,
                    'parent_agent_id': agent_id,
                    'subagent_type': subagent_type,
                    'prompt': prompt[:200],
                }

    def track_tool_result(self, request_id: int, tool_result_block: Dict[str, Any], timestamp: str = ""):
        """
        Track a tool_result block from request messages.

        Args:
            request_id: Request ID where result was received
            tool_result_block: Tool result content block
            timestamp: Timestamp of the request
        """
        tool_use_id = tool_result_block.get('tool_use_id')
        is_error = tool_result_block.get('is_error', False)

        if not tool_use_id:
            return

        agent_id = self.request_to_agent.get(request_id)
        if not agent_id:
            return

        # Store in index
        self.tool_result_index[tool_use_id] = {
            'tool_use_id': tool_use_id,
            'agent_id': agent_id,
            'request_id': request_id,
            'timestamp': timestamp,
            'is_error': is_error,
        }

        # Add to agent's tool_results
        agent = self.instances[agent_id]
        agent.tool_results.append({
            'tool_use_id': tool_use_id,
            'request_id': request_id,
            'timestamp': timestamp,
            'is_error': is_error,
        })

        # Create workflow edge: tool_use -> tool_result
        if tool_use_id in self.tool_use_index:
            tool_use_info = self.tool_use_index[tool_use_id]
            source_agent_id = tool_use_info['agent_id']
            target_agent_id = agent_id

            # Edge from agent that used tool to agent that received result
            self.workflow_edges.append({
                'type': 'tool_result',
                'source_agent_id': source_agent_id,
                'target_agent_id': target_agent_id,
                'tool_use_id': tool_use_id,
                'tool_name': tool_use_info['tool_name'],
                'source_request_id': tool_use_info['request_id'],
                'target_request_id': request_id,
                'is_error': is_error,
                'confidence': 1.0,  # Exact match via tool_use_id
            })

    def build_workflow_dag(self) -> Dict[str, Any]:
        """
        Build the complete workflow DAG showing branching and merging.

        Returns:
            Dictionary with nodes (agents) and edges (relationships)
        """
        nodes = []
        edges = []

        # Create nodes from agent instances
        for agent_id, agent in self.instances.items():
            node = {
                'id': agent_id,
                'agent_id': agent_id,
                'agent_type': agent.system_prompt_hash,
                'parent_agent_id': agent.parent_agent_id,
                'child_agent_ids': agent.child_agent_ids,
                'spawned_by_task_id': agent.spawned_by_task_id,
                'request_count': len(agent.requests),
                'tool_use_count': len(agent.tool_uses),
                'tool_result_count': len(agent.tool_results),
                'first_timestamp': agent.timestamps[0] if agent.timestamps else None,
                'last_timestamp': agent.timestamps[-1] if agent.timestamps else None,
                'is_root': agent.parent_agent_id is None,
                'is_leaf': len(agent.child_agent_ids) == 0,
            }
            nodes.append(node)

        # Add parent-child edges (subagent spawns)
        for agent_id, agent in self.instances.items():
            if agent.parent_agent_id:
                edges.append({
                    'type': 'subagent_spawn',
                    'source_agent_id': agent.parent_agent_id,
                    'target_agent_id': agent_id,
                    'spawned_by_task_id': agent.spawned_by_task_id,
                    'confidence': 0.95,
                })

        # Build request sequence edges
        self.build_request_sequence_edges()

        # Add all workflow edges (tool_result, content_reuse, request_sequence)
        edges.extend(self.workflow_edges)

        # Compute DAG metrics
        root_agents = [n for n in nodes if n['is_root']]
        leaf_agents = [n for n in nodes if n['is_leaf']]

        return {
            'nodes': nodes,
            'edges': edges,
            'metrics': {
                'total_agents': len(nodes),
                'root_agents': len(root_agents),
                'leaf_agents': len(leaf_agents),
                'total_edges': len(edges),
                'spawn_edges': len([e for e in edges if e['type'] == 'subagent_spawn']),
                'tool_result_edges': len([e for e in edges if e['type'] == 'tool_result']),
                'content_reuse_edges': len([e for e in edges if e['type'] == 'content_reuse']),
                'request_sequence_edges': len([e for e in edges if e['type'] == 'request_sequence']),
            },
            'root_agent_ids': [n['id'] for n in root_agents],
        }

    def get_agent_tree(self, root_agent_id: str, depth: int = 0) -> Dict[str, Any]:
        """
        Get tree structure starting from a root agent (for visualization).

        Args:
            root_agent_id: Agent ID to start from
            depth: Current depth (for recursion)

        Returns:
            Tree structure with agent and children
        """
        if root_agent_id not in self.instances:
            return None

        agent = self.instances[root_agent_id]

        tree = {
            'agent_id': agent.agent_id,
            'agent_type': agent.system_prompt_hash,
            'depth': depth,
            'request_count': len(agent.requests),
            'children': [],
        }

        # Recursively add children
        for child_id in agent.child_agent_ids:
            child_tree = self.get_agent_tree(child_id, depth + 1)
            if child_tree:
                tree['children'].append(child_tree)

        return tree

    def track_response_content(self, request_id: int, content_blocks: List[Any], timestamp: str = ""):
        """
        Track response content for content reuse detection.

        Args:
            request_id: Request ID that produced this content
            content_blocks: Response content blocks
            timestamp: Timestamp of the response
        """
        agent_id = self.request_to_agent.get(request_id)
        if not agent_id:
            return

        # Extract text content from response
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text', '')
                if text:
                    text_parts.append(text)

        if not text_parts:
            return

        # Hash first 200 chars of combined text
        combined_text = '\n'.join(text_parts)
        normalized = ' '.join(combined_text.split())[:200]
        content_hash = compute_hash(normalized, length=16)

        if content_hash not in self.response_content_index:
            self.response_content_index[content_hash] = []

        self.response_content_index[content_hash].append({
            'agent_id': agent_id,
            'request_id': request_id,
            'timestamp': timestamp,
            'content_hash': content_hash,
        })

    def track_request_content(self, request_id: int, messages: List[Dict], timestamp: str = ""):
        """
        Track request content and detect content reuse from previous responses.

        Args:
            request_id: Request ID being processed
            messages: Request messages
            timestamp: Timestamp of the request
        """
        agent_id = self.request_to_agent.get(request_id)
        if not agent_id:
            return

        # Extract text from user messages
        for message in messages:
            if not isinstance(message, dict):
                continue

            if message.get('role') != 'user':
                continue

            content = message.get('content', '')

            # Handle string content
            if isinstance(content, str):
                normalized = ' '.join(content.split())[:200]
                content_hash = compute_hash(normalized, length=16)

                if content_hash in self.response_content_index:
                    for source_info in self.response_content_index[content_hash]:
                        source_agent_id = source_info['agent_id']
                        source_request_id = source_info['request_id']

                        # Only create edge if source comes before target and different agents
                        if source_request_id < request_id and source_agent_id != agent_id:
                            self.workflow_edges.append({
                                'type': 'content_reuse',
                                'source_agent_id': source_agent_id,
                                'target_agent_id': agent_id,
                                'source_request_id': source_request_id,
                                'target_request_id': request_id,
                                'content_hash': content_hash,
                                'confidence': 0.85,
                            })

            # Handle array content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '')
                        if text:
                            normalized = ' '.join(text.split())[:200]
                            content_hash = compute_hash(normalized, length=16)

                            if content_hash in self.response_content_index:
                                for source_info in self.response_content_index[content_hash]:
                                    source_agent_id = source_info['agent_id']
                                    source_request_id = source_info['request_id']

                                    if source_request_id < request_id and source_agent_id != agent_id:
                                        self.workflow_edges.append({
                                            'type': 'content_reuse',
                                            'source_agent_id': source_agent_id,
                                            'target_agent_id': agent_id,
                                            'source_request_id': source_request_id,
                                            'target_request_id': request_id,
                                            'content_hash': content_hash,
                                            'confidence': 0.85,
                                        })

    def build_request_sequence_edges(self):
        """
        Build request sequence edges for each agent showing temporal flow.
        Creates edges between consecutive requests within the same agent.
        """
        from datetime import datetime

        for agent_id, agent in self.instances.items():
            if len(agent.requests) < 2:
                continue

            # Sort requests with timestamps
            request_times = []
            for i, req_id in enumerate(agent.requests):
                timestamp = agent.timestamps[i] if i < len(agent.timestamps) else None
                request_times.append((req_id, timestamp))

            # Create edges between consecutive requests
            for i in range(len(request_times) - 1):
                source_req_id, source_time = request_times[i]
                target_req_id, target_time = request_times[i + 1]

                # Calculate time gap
                time_gap_ms = None
                if source_time and target_time:
                    try:
                        source_dt = datetime.fromisoformat(source_time.replace('Z', '+00:00'))
                        target_dt = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
                        time_gap_ms = int((target_dt - source_dt).total_seconds() * 1000)
                    except (ValueError, AttributeError):
                        pass

                self.workflow_edges.append({
                    'type': 'request_sequence',
                    'source_agent_id': agent_id,
                    'target_agent_id': agent_id,
                    'source_request_id': source_req_id,
                    'target_request_id': target_req_id,
                    'time_gap_ms': time_gap_ms,
                    'confidence': 1.0,
                })

