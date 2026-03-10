"""
Conflict Detection Engine (CDE) - Real-time semantic conflict detection.

Analyzes the Semantic Intent Graph to detect three types of conflicts:
- Type 1: Contradictory Intent
- Type 2: Resource Contention  
- Type 3: Causal Violation

Uses hybrid approach: rule-based + embedding-based similarity.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from scf.core.sig import SemanticIntentGraph, IntentNode, IntentEdge, EdgeType, IntentStatus
from scf.core.pcl import ProcessContextLayer, ProcessSemanticModel


class ConflictType(Enum):
    CONTRADICTORY_INTENT = "type1_contradictory"
    RESOURCE_CONTENTION = "type2_resource"
    CAUSAL_VIOLATION = "type3_causal"


class ConflictSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectedConflict:
    """Represents a detected semantic conflict."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    agent_a_id: str
    agent_b_id: str
    intent_a_id: str
    intent_b_id: str
    entity: str
    description: str
    confidence: float  # 0.0 to 1.0
    detection_method: str  # "rule_based" or "embedding_based"
    detection_time_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ConflictDetectionEngine:
    """
    Detects semantic conflicts in the SIG using hybrid rule-based 
    and embedding-based analysis.
    """

    def __init__(self, pcl: ProcessContextLayer, use_embeddings: bool = True):
        self.pcl = pcl
        self.psm: ProcessSemanticModel = pcl.psm
        self.use_embeddings = use_embeddings
        self._embedding_model = None
        self._conflict_counter = 0
        self.detection_history: List[DetectedConflict] = []

        if use_embeddings:
            self._load_embedding_model()

    def _load_embedding_model(self):
        """Load the sentence-transformer model for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("Warning: sentence-transformers not installed. Using rule-based only.")
            self.use_embeddings = False

    def detect_conflicts(self, sig: SemanticIntentGraph) -> List[DetectedConflict]:
        """
        Run all conflict detection strategies on the current SIG.
        Returns list of newly detected conflicts.
        """
        start = time.time()
        conflicts = []

        # Compare pending intents against all active (pending + approved) intents
        pending = sig.get_pending_intents()
        has_process_model = len(self.psm.entity_models) > 0
        compare_against = [n for n in sig.nodes.values() 
                         if n.status in (IntentStatus.PENDING, IntentStatus.APPROVED)]
        
        seen_pairs = set()
        for intent_a in pending:
            for intent_b in compare_against:
                if intent_a.id == intent_b.id:  # Skip self
                    continue
                pair = frozenset([intent_a.id, intent_b.id])
                if pair in seen_pairs:  # Skip already checked pairs
                    continue
                seen_pairs.add(pair)
                # Type 1: Contradictory Intent
                t1_conflicts = self._detect_contradictory(intent_a, intent_b)
                conflicts.extend(t1_conflicts)

                # Type 2: Resource Contention
                t2_conflicts = self._detect_resource_contention(intent_a, intent_b)
                conflicts.extend(t2_conflicts)

                # Type 3: Causal Violation
                t3_conflicts = self._detect_causal_violation(intent_a, intent_b, sig)
                conflicts.extend(t3_conflicts)

        elapsed_ms = (time.time() - start) * 1000
        for c in conflicts:
            c.detection_time_ms = elapsed_ms / max(len(conflicts), 1)

        self.detection_history.extend(conflicts)
        return conflicts

    def _detect_contradictory(self, a: IntentNode, b: IntentNode) -> List[DetectedConflict]:
        """Type 1: Detect logically contradictory intents on shared entities."""
        conflicts = []
        shared = a.shares_entities_with(b)

        for entity in shared:
            post_a = a.postconditions.get(entity, "")
            post_b = b.postconditions.get(entity, "")

            if not post_a or not post_b:
                continue

            # Rule-based: check process model for mutual exclusivity
            entity_type = self._get_entity_type(entity)
            
            # WITH PROCESS MODEL: Full detection power
            if entity_type and entity_type in self.psm.entity_models:
                model = self.psm.entity_models[entity_type]
                resolved_a = model.resolve_synonym(post_a)
                resolved_b = model.resolve_synonym(post_b)
                
                # PCL ADVANTAGE 1: Synonym resolution — avoids false positives
                if resolved_a == resolved_b:
                    continue  # Same state after resolution → NOT a conflict
                
                # PCL ADVANTAGE 2: Mutual exclusivity knowledge — higher confidence
                if self.psm.check_state_exclusivity(entity_type, resolved_a, resolved_b):
                    self._conflict_counter += 1
                    conflicts.append(DetectedConflict(
                        conflict_id=f"C{self._conflict_counter:04d}",
                        conflict_type=ConflictType.CONTRADICTORY_INTENT,
                        severity=ConflictSeverity.CRITICAL,
                        agent_a_id=a.agent_id,
                        agent_b_id=b.agent_id,
                        intent_a_id=a.id,
                        intent_b_id=b.id,
                        entity=entity,
                        description=f"Contradictory states for '{entity}': '{resolved_a}' vs '{resolved_b}' (mutually exclusive)",
                        confidence=1.0,
                        detection_method="rule_based_pcl",
                    ))
                else:
                    # Different resolved states, not explicitly exclusive but concurrent = conflict
                    self._conflict_counter += 1
                    conflicts.append(DetectedConflict(
                        conflict_id=f"C{self._conflict_counter:04d}",
                        conflict_type=ConflictType.CONTRADICTORY_INTENT,
                        severity=ConflictSeverity.HIGH,
                        agent_a_id=a.agent_id,
                        agent_b_id=b.agent_id,
                        intent_a_id=a.id,
                        intent_b_id=b.id,
                        entity=entity,
                        description=f"Divergent states for '{entity}': '{resolved_a}' vs '{resolved_b}'",
                        confidence=0.85,
                        detection_method="rule_based_pcl",
                    ))
                continue
            
            # WITHOUT PROCESS MODEL: Weaker detection — no synonym resolution, no exclusivity knowledge
            # Can only compare raw strings, which produces false positives on synonyms
            if post_a != post_b:
                self._conflict_counter += 1
                conflicts.append(DetectedConflict(
                    conflict_id=f"C{self._conflict_counter:04d}",
                    conflict_type=ConflictType.CONTRADICTORY_INTENT,
                    severity=ConflictSeverity.MEDIUM,
                    agent_a_id=a.agent_id,
                    agent_b_id=b.agent_id,
                    intent_a_id=a.id,
                    intent_b_id=b.id,
                    entity=entity,
                    description=f"Different postconditions for '{entity}': '{post_a}' vs '{post_b}' (no process model — may be false positive)",
                    confidence=0.6,
                    detection_method="rule_based_nopcl",
                ))

        return conflicts

    def _detect_resource_contention(self, a: IntentNode, b: IntentNode) -> List[DetectedConflict]:
        """Type 2: Detect competing resource demands."""
        conflicts = []

        # Check if both agents require a shared precondition that one will invalidate
        for entity_a, required_a in a.preconditions.items():
            post_b = b.postconditions.get(entity_a)
            if post_b and post_b != required_a:
                # Agent B's action will invalidate Agent A's precondition
                entity_type = self._get_entity_type(entity_a)

                # Check if this is a real contention or a valid sequence
                temporal = self.psm.check_temporal_ordering(
                    a.action_type, b.action_type, entity_type
                ) if entity_type else None

                if temporal is None:  # No ordering defined = concurrent = contention
                    self._conflict_counter += 1
                    conflicts.append(DetectedConflict(
                        conflict_id=f"C{self._conflict_counter:04d}",
                        conflict_type=ConflictType.RESOURCE_CONTENTION,
                        severity=ConflictSeverity.HIGH,
                        agent_a_id=a.agent_id,
                        agent_b_id=b.agent_id,
                        intent_a_id=a.id,
                        intent_b_id=b.id,
                        entity=entity_a,
                        description=f"Resource contention on '{entity_a}': Agent {a.agent_id} requires '{required_a}', Agent {b.agent_id} will set to '{post_b}'",
                        confidence=0.85,
                        detection_method="rule_based",
                    ))

        # Check reverse direction
        for entity_b, required_b in b.preconditions.items():
            post_a = a.postconditions.get(entity_b)
            if post_a and post_a != required_b:
                # Avoid duplicates (check if already detected in forward pass)
                already_detected = any(
                    c.entity == entity_b and
                    c.conflict_type == ConflictType.RESOURCE_CONTENTION and
                    {c.agent_a_id, c.agent_b_id} == {a.agent_id, b.agent_id}
                    for c in conflicts
                )
                if not already_detected:
                    entity_type = self._get_entity_type(entity_b)
                    temporal = self.psm.check_temporal_ordering(
                        b.action_type, a.action_type, entity_type
                    ) if entity_type else None

                    if temporal is None:
                        self._conflict_counter += 1
                        conflicts.append(DetectedConflict(
                            conflict_id=f"C{self._conflict_counter:04d}",
                            conflict_type=ConflictType.RESOURCE_CONTENTION,
                            severity=ConflictSeverity.HIGH,
                            agent_a_id=b.agent_id,
                            agent_b_id=a.agent_id,
                            intent_a_id=b.id,
                            intent_b_id=a.id,
                            entity=entity_b,
                            description=f"Resource contention on '{entity_b}': Agent {b.agent_id} requires '{required_b}', Agent {a.agent_id} will set to '{post_a}'",
                            confidence=0.85,
                            detection_method="rule_based",
                        ))

        # Check numeric resource constraints
        conflicts.extend(self._check_numeric_contention(a, b))

        return conflicts

    def _check_numeric_contention(self, a: IntentNode, b: IntentNode) -> List[DetectedConflict]:
        """Check for numeric resource over-allocation."""
        conflicts = []

        # Look for numeric values in metadata that represent resource consumption
        a_consumption = a.metadata.get("resource_consumption", {})
        b_consumption = b.metadata.get("resource_consumption", {})

        for resource, amount_a in a_consumption.items():
            amount_b = b_consumption.get(resource, 0)
            if amount_a + amount_b > 0:
                available = a.metadata.get("resource_available", {}).get(resource)
                if available is not None and amount_a + amount_b > available:
                    self._conflict_counter += 1
                    conflicts.append(DetectedConflict(
                        conflict_id=f"C{self._conflict_counter:04d}",
                        conflict_type=ConflictType.RESOURCE_CONTENTION,
                        severity=ConflictSeverity.CRITICAL,
                        agent_a_id=a.agent_id,
                        agent_b_id=b.agent_id,
                        intent_a_id=a.id,
                        intent_b_id=b.id,
                        entity=resource,
                        description=f"Over-allocation of '{resource}': {amount_a} + {amount_b} = {amount_a + amount_b} > {available} available",
                        confidence=1.0,
                        detection_method="rule_based",
                    ))

        return conflicts

    def _detect_causal_violation(self, a: IntentNode, b: IntentNode, sig: SemanticIntentGraph) -> List[DetectedConflict]:
        """Type 3: Detect causal dependency violations."""
        conflicts = []

        # Check if B depends on a state that A will alter
        for entity, required_state in b.preconditions.items():
            a_post = a.postconditions.get(entity)
            if a_post and a_post != required_state:
                entity_type = self._get_entity_type(entity)

                # Check temporal ordering
                temporal = self.psm.check_temporal_ordering(
                    a.action_type, b.action_type, entity_type
                ) if entity_type else None

                if temporal is True:
                    # A MUST happen before B, but A will break B's precondition
                    self._conflict_counter += 1
                    conflicts.append(DetectedConflict(
                        conflict_id=f"C{self._conflict_counter:04d}",
                        conflict_type=ConflictType.CAUSAL_VIOLATION,
                        severity=ConflictSeverity.HIGH,
                        agent_a_id=a.agent_id,
                        agent_b_id=b.agent_id,
                        intent_a_id=a.id,
                        intent_b_id=b.id,
                        entity=entity,
                        description=f"Causal violation: Agent {a.agent_id} will change '{entity}' to '{a_post}' (required before), breaking Agent {b.agent_id}'s precondition of '{required_state}'",
                        confidence=0.9,
                        detection_method="rule_based",
                    ))

        # Check explicit dependencies in the SIG
        for dep_id in b.dependencies:
            if dep_id == a.id:
                # B depends on A, check if A's postconditions satisfy B's needs
                for entity, required in b.preconditions.items():
                    a_provides = a.postconditions.get(entity)
                    if a_provides and a_provides != required:
                        self._conflict_counter += 1
                        conflicts.append(DetectedConflict(
                            conflict_id=f"C{self._conflict_counter:04d}",
                            conflict_type=ConflictType.CAUSAL_VIOLATION,
                            severity=ConflictSeverity.CRITICAL,
                            agent_a_id=a.agent_id,
                            agent_b_id=b.agent_id,
                            intent_a_id=a.id,
                            intent_b_id=b.id,
                            entity=entity,
                            description=f"Dependency violation: {b.id} depends on {a.id} for '{entity}'='{required}', but {a.id} will produce '{a_provides}'",
                            confidence=1.0,
                            detection_method="rule_based",
                        ))

        return conflicts

    def _compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two text descriptions."""
        if not self._embedding_model:
            return 0.5  # Neutral if no model
        try:
            embeddings = self._embedding_model.encode([text_a, text_b])
            from numpy import dot
            from numpy.linalg import norm
            return float(dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1])))
        except Exception:
            return 0.5

    def _get_entity_type(self, entity: str) -> Optional[str]:
        """Extract entity type from entity identifier."""
        # Convention: entity IDs are like "order_123", "ticket_456"
        parts = entity.split("_")
        if len(parts) >= 1:
            candidate = parts[0]
            if candidate in self.psm.entity_models:
                return candidate
        # Try full entity string
        if entity in self.psm.entity_models:
            return entity
        return None

    def get_detection_stats(self) -> Dict:
        """Return detection statistics."""
        total = len(self.detection_history)
        if total == 0:
            return {"total": 0}
        return {
            "total": total,
            "type1_contradictory": len([c for c in self.detection_history if c.conflict_type == ConflictType.CONTRADICTORY_INTENT]),
            "type2_resource": len([c for c in self.detection_history if c.conflict_type == ConflictType.RESOURCE_CONTENTION]),
            "type3_causal": len([c for c in self.detection_history if c.conflict_type == ConflictType.CAUSAL_VIOLATION]),
            "rule_based": len([c for c in self.detection_history if c.detection_method == "rule_based"]),
            "embedding_based": len([c for c in self.detection_history if c.detection_method == "embedding_based"]),
            "avg_confidence": sum(c.confidence for c in self.detection_history) / total,
            "avg_detection_time_ms": sum(c.detection_time_ms for c in self.detection_history) / total,
        }
