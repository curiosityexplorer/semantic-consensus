"""
Semantic Intent Graph (SIG) - Core data structure for the SCF framework.

The SIG captures what agents INTEND to do before they do it,
enabling conflict detection before actions are committed.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


class EdgeType(Enum):
    DEPENDENCY = "dependency"
    POTENTIAL_CONFLICT = "potential_conflict"
    CAUSAL_CHAIN = "causal_chain"


class IntentStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    BLOCKED = "blocked"
    RESOLVED = "resolved"
    EXECUTED = "executed"
    ESCALATED = "escalated"


@dataclass
class IntentNode:
    """Represents a single agent's intent to perform an action."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    agent_role: str = ""
    action_type: str = ""  # Using PCL shared vocabulary
    target_entities: List[str] = field(default_factory=list)
    preconditions: Dict[str, str] = field(default_factory=dict)  # entity -> required_state
    postconditions: Dict[str, str] = field(default_factory=dict)  # entity -> resulting_state
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    status: IntentStatus = IntentStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # IDs of intent nodes this depends on
    metadata: Dict = field(default_factory=dict)

    def affects_entity(self, entity: str) -> bool:
        return entity in self.target_entities

    def shares_entities_with(self, other: 'IntentNode') -> Set[str]:
        return set(self.target_entities) & set(other.target_entities)


@dataclass
class IntentEdge:
    """Represents a relationship between two intent nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    description: str = ""
    conflict_score: float = 0.0  # 0.0 = no conflict, 1.0 = definite conflict


class SemanticIntentGraph:
    """
    Directed labeled graph that captures agent intents and their relationships.
    Central data structure of the Semantic Consensus Framework.
    """

    def __init__(self):
        self.nodes: Dict[str, IntentNode] = {}
        self.edges: List[IntentEdge] = []
        self._entity_index: Dict[str, List[str]] = {}  # entity -> [node_ids]
        self._registration_times: List[Tuple[float, str]] = []  # (timestamp, node_id)

    def register_intent(self, intent: IntentNode) -> str:
        """
        Register a new intent node in the SIG.
        Automatically constructs edges to existing related nodes.
        Returns the intent node ID.
        """
        self.nodes[intent.id] = intent
        self._registration_times.append((intent.timestamp, intent.id))

        # Update entity index
        for entity in intent.target_entities:
            if entity not in self._entity_index:
                self._entity_index[entity] = []
            self._entity_index[entity].append(intent.id)

        # Auto-construct edges to existing nodes
        self._build_edges_for(intent)

        return intent.id

    def _build_edges_for(self, new_node: IntentNode):
        """Automatically build edges between new node and existing nodes."""
        for existing_id, existing_node in self.nodes.items():
            if existing_id == new_node.id:
                continue
            if existing_node.status in (IntentStatus.EXECUTED, IntentStatus.BLOCKED):
                continue

            shared = new_node.shares_entities_with(existing_node)
            if not shared:
                continue

            # Check for potential conflict (overlapping entities with different postconditions)
            for entity in shared:
                new_post = new_node.postconditions.get(entity)
                existing_post = existing_node.postconditions.get(entity)

                if new_post and existing_post and new_post != existing_post:
                    self.edges.append(IntentEdge(
                        source_id=existing_id,
                        target_id=new_node.id,
                        edge_type=EdgeType.POTENTIAL_CONFLICT,
                        description=f"Entity '{entity}': '{existing_post}' vs '{new_post}'",
                        conflict_score=0.8
                    ))

            # Check for dependency (new node requires existing node's postcondition)
            for entity, required_state in new_node.preconditions.items():
                existing_post = existing_node.postconditions.get(entity)
                if existing_post == required_state:
                    self.edges.append(IntentEdge(
                        source_id=existing_id,
                        target_id=new_node.id,
                        edge_type=EdgeType.DEPENDENCY,
                        description=f"Node {new_node.id} depends on {existing_id} for '{entity}' = '{required_state}'"
                    ))

            # Check for causal chain (existing node will alter state new node assumes)
            for entity, required_state in new_node.preconditions.items():
                existing_post = existing_node.postconditions.get(entity)
                if existing_post and existing_post != required_state:
                    self.edges.append(IntentEdge(
                        source_id=existing_id,
                        target_id=new_node.id,
                        edge_type=EdgeType.CAUSAL_CHAIN,
                        description=f"Node {existing_id} will change '{entity}' to '{existing_post}', breaking {new_node.id}'s precondition '{required_state}'",
                        conflict_score=0.7
                    ))

    def get_pending_intents(self) -> List[IntentNode]:
        """Get all intents that haven't been resolved yet."""
        return [n for n in self.nodes.values() if n.status == IntentStatus.PENDING]

    def get_conflict_edges(self) -> List[IntentEdge]:
        """Get all edges that represent potential conflicts."""
        return [e for e in self.edges if e.edge_type == EdgeType.POTENTIAL_CONFLICT]

    def get_causal_edges(self) -> List[IntentEdge]:
        """Get all edges representing causal chain risks."""
        return [e for e in self.edges if e.edge_type == EdgeType.CAUSAL_CHAIN]

    def get_intents_for_entity(self, entity: str) -> List[IntentNode]:
        """Get all intent nodes that affect a given entity."""
        node_ids = self._entity_index.get(entity, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def update_status(self, node_id: str, status: IntentStatus):
        """Update the status of an intent node."""
        if node_id in self.nodes:
            self.nodes[node_id].status = status

    def get_temporal_order(self, node_id_a: str, node_id_b: str) -> Optional[str]:
        """Returns ID of the node that was registered first."""
        if node_id_a in self.nodes and node_id_b in self.nodes:
            if self.nodes[node_id_a].timestamp <= self.nodes[node_id_b].timestamp:
                return node_id_a
            return node_id_b
        return None

    def clear_executed(self):
        """Remove executed intents to keep the graph manageable."""
        executed = [nid for nid, n in self.nodes.items() if n.status == IntentStatus.EXECUTED]
        for nid in executed:
            # Remove from entity index
            for entity in self.nodes[nid].target_entities:
                if entity in self._entity_index:
                    self._entity_index[entity] = [
                        x for x in self._entity_index[entity] if x != nid
                    ]
            del self.nodes[nid]
        # Remove edges referencing deleted nodes
        self.edges = [
            e for e in self.edges
            if e.source_id in self.nodes and e.target_id in self.nodes
        ]

    def get_stats(self) -> Dict:
        """Return graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "pending_nodes": len([n for n in self.nodes.values() if n.status == IntentStatus.PENDING]),
            "conflict_edges": len(self.get_conflict_edges()),
            "causal_edges": len(self.get_causal_edges()),
            "total_edges": len(self.edges),
            "tracked_entities": len(self._entity_index),
        }
