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
            'spawned_by_task_id': self.spawned_by_task_id,
            'parent_agent_id': self.parent_agent_id,
            'first_user_message': self.first_user_message[:200],  # Truncate for readability
            'total_requests': len(self.requests),
            'conversation_turns': len(self.message_count_history),
        }


def compute_conversation_fingerprint(messages: List[Dict]) -> str:
    """
    Compute a unique fingerprint for a conversation based on message sequence.
    
    This creates a hash representing the conversation state. Two requests with
    identical conversation history will have the same fingerprint.
    
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
        
        # Create a signature for this message
        if isinstance(content, str):
            # Simple text message - use role + hash of content
            content_hash = compute_hash(content, length=8)
            fingerprint_parts.append(f"{role}:text:{content_hash}")
        
        elif isinstance(content, list):
            # Structured content - use role + block types + IDs
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
                        # For tool_use, use tool_use_id (unique identifier)
                        tool_use_id = block.get('id', '')
                        tool_name = block.get('name', '')
                        block_signature.append(f"tool_use:{tool_name}:{tool_use_id}")
                    
                    elif block_type == 'tool_result':
                        # For tool_result, use tool_use_id it references
                        tool_use_id = block.get('tool_use_id', '')
                        block_signature.append(f"tool_result:{tool_use_id}")
            
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

    def identify_or_create_agent(self, request_id: int, body: Dict) -> AgentInstance:
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
        )

        # Check if spawned by Task tool
        spawning_info = self.detect_task_spawn(first_user_msg)
        if spawning_info:
            agent.spawned_by_task_id = spawning_info['task_id']
            agent.parent_agent_id = spawning_info['parent_agent_id']

        self.instances[agent_id] = agent
        self.fingerprint_to_agent[conversation_fingerprint] = agent_id
        self.request_to_agent[request_id] = agent_id

        return agent

    def find_parent_conversation(self, messages: List[Dict], system_prompt_hash: str) -> Optional[AgentInstance]:
        """
        Find if this conversation is a continuation of an existing agent.

        Strategy: Check if the first N-1 messages match an existing agent's conversation.

        Args:
            messages: Current message list
            system_prompt_hash: System prompt hash for this request

        Returns:
            AgentInstance if parent found, None otherwise
        """
        if len(messages) <= 1:
            return None

        # Compute fingerprint of conversation minus last message
        parent_fingerprint = compute_conversation_fingerprint(messages[:-1])

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

