#!/usr/bin/env python3
"""
Entity Deduplication for Claude Code workflow logs.

Tracks entity occurrences across multiple requests and marks duplicates
caused by conversation history replay.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict


class EntityDeduplicator:
    """Deduplicate entities based on agent instance tracking."""
    
    def __init__(self, agent_tracker):
        """
        Initialize deduplicator.
        
        Args:
            agent_tracker: AgentInstanceTracker instance for agent context
        """
        self.agent_tracker = agent_tracker
        self.unique_entities: Dict[str, Dict] = {}  # entity_id -> entity with metadata
        self.entity_type_counts: Dict[str, int] = defaultdict(int)
        self.entity_type_duplicates: Dict[str, int] = defaultdict(int)
    
    def deduplicate_entity(self, entity: Dict, entity_type: str, request_id: int) -> Dict:
        """
        Mark entity as duplicate or unique.
        
        Args:
            entity: Entity dictionary with 'id' field
            entity_type: Type of entity (e.g., 'tool_use', 'task', 'message')
            request_id: Request ID where this entity was seen
            
        Returns:
            Enriched entity with deduplication metadata
        """
        entity_id = entity.get('id')
        if not entity_id:
            # No ID, cannot deduplicate - treat as unique
            return {
                **entity,
                'is_duplicate': False,
                'entity_type': entity_type,
            }
        
        agent_id = self.agent_tracker.request_to_agent.get(request_id)
        
        if entity_id not in self.unique_entities:
            # First occurrence
            enriched = {
                **entity,
                'entity_type': entity_type,
                'is_duplicate': False,
                'first_seen_request': request_id,
                'first_seen_agent': agent_id,
                'occurrence_count': 1,
                'seen_in_requests': [request_id],
                'seen_in_agents': [agent_id] if agent_id else [],
            }
            
            self.unique_entities[entity_id] = enriched
            self.entity_type_counts[entity_type] += 1
            return enriched
        
        else:
            # Duplicate occurrence
            unique_entity = self.unique_entities[entity_id]
            unique_entity['occurrence_count'] += 1
            unique_entity['seen_in_requests'].append(request_id)
            
            if agent_id and agent_id not in unique_entity['seen_in_agents']:
                unique_entity['seen_in_agents'].append(agent_id)
            
            self.entity_type_duplicates[entity_type] += 1
            
            return {
                **entity,
                'entity_type': entity_type,
                'is_duplicate': True,
                'duplicate_of': entity_id,
                'first_seen_request': unique_entity['first_seen_request'],
                'first_seen_agent': unique_entity['first_seen_agent'],
                'occurrence_count': unique_entity['occurrence_count'],
            }
    
    def get_unique_entities_only(self, entity_type: Optional[str] = None) -> List[Dict]:
        """
        Return only unique entities (filter out duplicates).
        
        Args:
            entity_type: Optional filter by entity type
            
        Returns:
            List of unique entities
        """
        entities = list(self.unique_entities.values())
        
        if entity_type:
            entities = [e for e in entities if e.get('entity_type') == entity_type]
        
        return entities
    
    def get_deduplication_stats(self) -> Dict:
        """Get statistics on deduplication."""
        total_unique = len(self.unique_entities)
        total_occurrences = sum(e['occurrence_count'] for e in self.unique_entities.values())
        
        # Stats by entity type
        type_stats = {}
        for entity_type in self.entity_type_counts.keys():
            unique_count = self.entity_type_counts[entity_type]
            duplicate_count = self.entity_type_duplicates.get(entity_type, 0)
            total_count = unique_count + duplicate_count
            
            type_stats[entity_type] = {
                'unique': unique_count,
                'duplicates': duplicate_count,
                'total': total_count,
                'duplication_ratio': total_count / unique_count if unique_count > 0 else 0,
            }
        
        return {
            'total_unique_entities': total_unique,
            'total_occurrences': total_occurrences,
            'overall_duplication_ratio': total_occurrences / total_unique if total_unique > 0 else 0,
            'duplicates_removed': total_occurrences - total_unique,
            'by_entity_type': type_stats,
        }
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """Get unique entity by ID."""
        return self.unique_entities.get(entity_id)
    
    def get_entities_by_agent(self, agent_id: str) -> List[Dict]:
        """Get all unique entities first seen by a specific agent."""
        return [
            e for e in self.unique_entities.values()
            if e.get('first_seen_agent') == agent_id
        ]
    
    def get_cross_agent_entities(self) -> List[Dict]:
        """Get entities that appear in multiple agents (shared entities)."""
        return [
            e for e in self.unique_entities.values()
            if len(e.get('seen_in_agents', [])) > 1
        ]

