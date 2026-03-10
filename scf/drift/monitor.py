"""
Drift Monitor (DM) - Continuous semantic divergence detection.

Monitors agent context alignment over long-running workflows and
triggers proactive re-synchronization when drift exceeds threshold.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scf.core.sig import SemanticIntentGraph, IntentNode, IntentStatus
from scf.core.pcl import ProcessContextLayer


@dataclass
class AgentState:
    """Snapshot of an agent's current working state."""
    agent_id: str
    agent_role: str
    current_entity_states: Dict[str, str] = field(default_factory=dict)
    planned_actions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    interaction_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DriftEvent:
    """Recorded when drift is detected between agents."""
    event_id: str
    agent_a_id: str
    agent_b_id: str
    sas_score: float  # Semantic Alignment Score
    threshold: float
    entities_diverged: List[str]
    interaction_step: int
    description: str
    resync_triggered: bool = False
    conflict_prevented: bool = False  # Set to True if a conflict would have occurred
    timestamp: float = field(default_factory=time.time)


class DriftMonitor:
    """
    Continuously monitors semantic alignment between cooperating agents.
    Triggers re-synchronization when alignment drops below threshold.
    """

    def __init__(self, pcl: ProcessContextLayer, threshold: float = 0.6):
        self.pcl = pcl
        self.threshold = threshold
        self.agent_states: Dict[str, AgentState] = {}
        self.drift_events: List[DriftEvent] = []
        self._event_counter = 0
        self._sas_history: Dict[str, List[Tuple[int, float]]] = {}  # pair_key -> [(step, sas)]

    def update_agent_state(self, state: AgentState):
        """Update the monitored state for an agent."""
        self.agent_states[state.agent_id] = state

    def check_alignment(self, sig: SemanticIntentGraph, interaction_step: int) -> List[DriftEvent]:
        """
        Check semantic alignment between all pairs of active agents.
        Returns list of drift events for pairs below threshold.
        """
        events = []
        agents = list(self.agent_states.values())

        for i, state_a in enumerate(agents):
            for state_b in agents[i + 1:]:
                sas = self._compute_sas(state_a, state_b, sig)
                pair_key = f"{state_a.agent_id}:{state_b.agent_id}"

                # Track history
                if pair_key not in self._sas_history:
                    self._sas_history[pair_key] = []
                self._sas_history[pair_key].append((interaction_step, sas))

                if sas < self.threshold:
                    self._event_counter += 1
                    diverged = self._find_diverged_entities(state_a, state_b)
                    event = DriftEvent(
                        event_id=f"D{self._event_counter:04d}",
                        agent_a_id=state_a.agent_id,
                        agent_b_id=state_b.agent_id,
                        sas_score=sas,
                        threshold=self.threshold,
                        entities_diverged=diverged,
                        interaction_step=interaction_step,
                        description=f"Semantic drift detected: SAS={sas:.3f} < {self.threshold} between {state_a.agent_id} and {state_b.agent_id}",
                        resync_triggered=True,
                    )
                    events.append(event)
                    self.drift_events.append(event)

        return events

    def _compute_sas(self, state_a: AgentState, state_b: AgentState, sig: SemanticIntentGraph) -> float:
        """
        Compute Semantic Alignment Score between two agents.
        SAS = weighted average of:
          (a) Entity state overlap (0-1)
          (b) Action consistency with process model (0-1)
          (c) Confidence alignment (0-1)
        """
        # (a) Entity state overlap
        entity_score = self._entity_state_overlap(state_a, state_b)

        # (b) Action consistency - check if planned actions are valid transitions
        action_score_a = self._action_consistency(state_a)
        action_score_b = self._action_consistency(state_b)
        action_score = (action_score_a + action_score_b) / 2

        # (c) Confidence alignment
        conf_diff = abs(state_a.confidence - state_b.confidence)
        confidence_score = 1.0 - conf_diff

        # Weighted combination
        sas = 0.5 * entity_score + 0.3 * action_score + 0.2 * confidence_score
        return sas

    def _entity_state_overlap(self, a: AgentState, b: AgentState) -> float:
        """Compute overlap between agents' entity state models."""
        all_entities = set(a.current_entity_states.keys()) | set(b.current_entity_states.keys())
        if not all_entities:
            return 1.0

        matching = 0
        for entity in all_entities:
            state_a = a.current_entity_states.get(entity)
            state_b = b.current_entity_states.get(entity)

            if state_a is None or state_b is None:
                # One agent doesn't track this entity - partial penalty
                matching += 0.5
            elif self.pcl.psm.resolve_term(state_a) == self.pcl.psm.resolve_term(state_b):
                matching += 1.0
            # else: 0 for mismatch

        return matching / len(all_entities)

    def _action_consistency(self, state: AgentState) -> float:
        """Check if an agent's planned actions are valid in the process model."""
        if not state.planned_actions:
            return 1.0

        valid = 0
        total = 0
        for action in state.planned_actions:
            # Parse action as "entity:from_state->to_state"
            parts = action.split(":")
            if len(parts) == 2:
                entity = parts[0]
                transition = parts[1].split("->")
                if len(transition) == 2:
                    total += 1
                    entity_type = entity.split("_")[0] if "_" in entity else entity
                    if self.pcl.psm.check_valid_transition(entity_type, transition[0], transition[1]):
                        valid += 1

        return valid / max(total, 1)

    def _find_diverged_entities(self, a: AgentState, b: AgentState) -> List[str]:
        """Find entities where agents disagree on state."""
        diverged = []
        all_entities = set(a.current_entity_states.keys()) | set(b.current_entity_states.keys())
        for entity in all_entities:
            state_a = a.current_entity_states.get(entity)
            state_b = b.current_entity_states.get(entity)
            if state_a and state_b:
                if self.pcl.psm.resolve_term(state_a) != self.pcl.psm.resolve_term(state_b):
                    diverged.append(entity)
        return diverged

    def get_resync_context(self, agent_id: str, entity_types: List[str]) -> Dict:
        """Generate re-synchronization context for a drifting agent."""
        return self.pcl.get_agent_context(
            self.agent_states[agent_id].agent_role if agent_id in self.agent_states else "unknown",
            entity_types
        )

    def get_drift_stats(self) -> Dict:
        """Return drift monitoring statistics."""
        total = len(self.drift_events)
        if total == 0:
            return {"total": 0}
        return {
            "total_drift_events": total,
            "resyncs_triggered": len([e for e in self.drift_events if e.resync_triggered]),
            "conflicts_prevented": len([e for e in self.drift_events if e.conflict_prevented]),
            "avg_sas_at_drift": sum(e.sas_score for e in self.drift_events) / total,
            "false_alarms": len([e for e in self.drift_events if e.resync_triggered and not e.conflict_prevented]),
        }
