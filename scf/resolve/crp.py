"""
Consensus Resolution Protocol (CRP) - Principled conflict resolution.

Resolves conflicts through a three-tier hierarchy:
  Tier 1: Policy Authority (highest)
  Tier 2: Capability Authority
  Tier 3: Temporal Priority (lowest)
  Escalation: Human review
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from scf.core.sig import SemanticIntentGraph, IntentNode, IntentStatus
from scf.core.pcl import ProcessContextLayer
from scf.detect.cde import DetectedConflict, ConflictType


class ResolutionTier(Enum):
    POLICY = "tier1_policy"
    CAPABILITY = "tier2_capability"
    TEMPORAL = "tier3_temporal"
    ESCALATION = "escalation"


class ResolutionOutcome(Enum):
    AGENT_A_WINS = "agent_a_wins"
    AGENT_B_WINS = "agent_b_wins"
    BOTH_BLOCKED = "both_blocked"
    ESCALATED = "escalated"
    MERGED = "merged"  # Both can proceed with modifications


@dataclass
class ResolutionDecision:
    """The result of resolving a conflict."""
    conflict_id: str
    resolution_tier: ResolutionTier
    outcome: ResolutionOutcome
    winner_intent_id: Optional[str] = None
    loser_intent_id: Optional[str] = None
    reason: str = ""
    resolution_time_ms: float = 0.0
    policy_applied: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PolicyRule:
    """A governance policy rule for conflict resolution."""
    rule_id: str
    description: str
    entity_type: str
    condition: str  # "compliance_vs_any", "risk_vs_convenience", etc.
    winner_role: str  # Role that wins when this rule applies
    priority: int = 100


class ConsensusResolutionProtocol:
    """Resolves semantic conflicts through a principled three-tier hierarchy."""

    def __init__(self, pcl: ProcessContextLayer):
        self.pcl = pcl
        self.policy_rules: List[PolicyRule] = []
        self.resolution_history: List[ResolutionDecision] = []
        self._role_authority: Dict[str, int] = {}  # role -> authority_level

    def add_policy_rule(self, rule: PolicyRule):
        """Add a governance policy rule."""
        self.policy_rules.append(rule)

    def set_role_authority(self, role: str, authority_level: int):
        """Set the authority level for a role (used in Tier 2)."""
        self._role_authority[role] = authority_level

    def resolve(self, conflict: DetectedConflict, sig: SemanticIntentGraph) -> ResolutionDecision:
        """
        Resolve a detected conflict using the three-tier hierarchy.
        """
        start = time.time()

        intent_a = sig.nodes.get(conflict.intent_a_id)
        intent_b = sig.nodes.get(conflict.intent_b_id)

        if not intent_a or not intent_b:
            return ResolutionDecision(
                conflict_id=conflict.conflict_id,
                resolution_tier=ResolutionTier.ESCALATION,
                outcome=ResolutionOutcome.ESCALATED,
                reason="One or both intents no longer exist in SIG"
            )

        # Tier 1: Policy Authority
        decision = self._resolve_by_policy(conflict, intent_a, intent_b)
        if decision:
            decision.resolution_time_ms = (time.time() - start) * 1000
            self._apply_decision(decision, sig)
            self.resolution_history.append(decision)
            return decision

        # Tier 2: Capability Authority
        decision = self._resolve_by_capability(conflict, intent_a, intent_b)
        if decision:
            decision.resolution_time_ms = (time.time() - start) * 1000
            self._apply_decision(decision, sig)
            self.resolution_history.append(decision)
            return decision

        # Tier 3: Temporal Priority
        decision = self._resolve_by_temporal(conflict, intent_a, intent_b, sig)
        if decision:
            decision.resolution_time_ms = (time.time() - start) * 1000
            self._apply_decision(decision, sig)
            self.resolution_history.append(decision)
            return decision

        # Escalation
        decision = ResolutionDecision(
            conflict_id=conflict.conflict_id,
            resolution_tier=ResolutionTier.ESCALATION,
            outcome=ResolutionOutcome.ESCALATED,
            reason=f"Could not resolve conflict through any tier. Agents: {conflict.agent_a_id} vs {conflict.agent_b_id}",
        )
        decision.resolution_time_ms = (time.time() - start) * 1000
        self._apply_decision(decision, sig)
        self.resolution_history.append(decision)
        return decision

    def _resolve_by_policy(self, conflict: DetectedConflict, a: IntentNode, b: IntentNode) -> Optional[ResolutionDecision]:
        """Tier 1: Check governance policies."""
        entity_type = conflict.entity.split("_")[0] if "_" in conflict.entity else conflict.entity

        for rule in sorted(self.policy_rules, key=lambda r: -r.priority):
            if rule.entity_type in ("*", entity_type):
                # Check if either agent's role matches the winner role
                if a.agent_role == rule.winner_role:
                    return ResolutionDecision(
                        conflict_id=conflict.conflict_id,
                        resolution_tier=ResolutionTier.POLICY,
                        outcome=ResolutionOutcome.AGENT_A_WINS,
                        winner_intent_id=a.id,
                        loser_intent_id=b.id,
                        reason=f"Policy '{rule.description}': role '{a.agent_role}' takes precedence",
                        policy_applied=rule.rule_id,
                    )
                elif b.agent_role == rule.winner_role:
                    return ResolutionDecision(
                        conflict_id=conflict.conflict_id,
                        resolution_tier=ResolutionTier.POLICY,
                        outcome=ResolutionOutcome.AGENT_B_WINS,
                        winner_intent_id=b.id,
                        loser_intent_id=a.id,
                        reason=f"Policy '{rule.description}': role '{b.agent_role}' takes precedence",
                        policy_applied=rule.rule_id,
                    )

        return None

    def _resolve_by_capability(self, conflict: DetectedConflict, a: IntentNode, b: IntentNode) -> Optional[ResolutionDecision]:
        """Tier 2: Check capability/authority levels."""
        auth_a = self._role_authority.get(a.agent_role, 0)
        auth_b = self._role_authority.get(b.agent_role, 0)

        # Also check PCL authority mappings
        entity_type = conflict.entity.split("_")[0] if "_" in conflict.entity else conflict.entity
        pcl_auth_a = self.pcl.psm.get_role_authority(a.agent_role, entity_type)
        pcl_auth_b = self.pcl.psm.get_role_authority(b.agent_role, entity_type)

        total_a = auth_a + pcl_auth_a
        total_b = auth_b + pcl_auth_b

        if total_a > total_b:
            return ResolutionDecision(
                conflict_id=conflict.conflict_id,
                resolution_tier=ResolutionTier.CAPABILITY,
                outcome=ResolutionOutcome.AGENT_A_WINS,
                winner_intent_id=a.id,
                loser_intent_id=b.id,
                reason=f"Capability authority: '{a.agent_role}' (auth={total_a}) > '{b.agent_role}' (auth={total_b})",
            )
        elif total_b > total_a:
            return ResolutionDecision(
                conflict_id=conflict.conflict_id,
                resolution_tier=ResolutionTier.CAPABILITY,
                outcome=ResolutionOutcome.AGENT_B_WINS,
                winner_intent_id=b.id,
                loser_intent_id=a.id,
                reason=f"Capability authority: '{b.agent_role}' (auth={total_b}) > '{a.agent_role}' (auth={total_a})",
            )

        # Equal authority — fall through to Tier 3
        return None

    def _resolve_by_temporal(self, conflict: DetectedConflict, a: IntentNode, b: IntentNode, sig: SemanticIntentGraph) -> Optional[ResolutionDecision]:
        """Tier 3: First-registered intent wins."""
        first_id = sig.get_temporal_order(a.id, b.id)
        if first_id:
            if first_id == a.id:
                return ResolutionDecision(
                    conflict_id=conflict.conflict_id,
                    resolution_tier=ResolutionTier.TEMPORAL,
                    outcome=ResolutionOutcome.AGENT_A_WINS,
                    winner_intent_id=a.id,
                    loser_intent_id=b.id,
                    reason=f"Temporal priority: intent {a.id} registered before {b.id}",
                )
            else:
                return ResolutionDecision(
                    conflict_id=conflict.conflict_id,
                    resolution_tier=ResolutionTier.TEMPORAL,
                    outcome=ResolutionOutcome.AGENT_B_WINS,
                    winner_intent_id=b.id,
                    loser_intent_id=a.id,
                    reason=f"Temporal priority: intent {b.id} registered before {a.id}",
                )
        return None

    def _apply_decision(self, decision: ResolutionDecision, sig: SemanticIntentGraph):
        """Apply a resolution decision to the SIG."""
        if decision.outcome == ResolutionOutcome.AGENT_A_WINS:
            sig.update_status(decision.winner_intent_id, IntentStatus.APPROVED)
            sig.update_status(decision.loser_intent_id, IntentStatus.BLOCKED)
        elif decision.outcome == ResolutionOutcome.AGENT_B_WINS:
            sig.update_status(decision.winner_intent_id, IntentStatus.APPROVED)
            sig.update_status(decision.loser_intent_id, IntentStatus.BLOCKED)
        elif decision.outcome == ResolutionOutcome.BOTH_BLOCKED:
            if decision.winner_intent_id:
                sig.update_status(decision.winner_intent_id, IntentStatus.BLOCKED)
            if decision.loser_intent_id:
                sig.update_status(decision.loser_intent_id, IntentStatus.BLOCKED)
        elif decision.outcome == ResolutionOutcome.ESCALATED:
            # Keep both as pending for human review
            pass

    def get_resolution_stats(self) -> Dict:
        """Return resolution statistics."""
        total = len(self.resolution_history)
        if total == 0:
            return {"total": 0}
        return {
            "total": total,
            "tier1_policy": len([d for d in self.resolution_history if d.resolution_tier == ResolutionTier.POLICY]),
            "tier2_capability": len([d for d in self.resolution_history if d.resolution_tier == ResolutionTier.CAPABILITY]),
            "tier3_temporal": len([d for d in self.resolution_history if d.resolution_tier == ResolutionTier.TEMPORAL]),
            "escalated": len([d for d in self.resolution_history if d.resolution_tier == ResolutionTier.ESCALATION]),
            "avg_resolution_time_ms": sum(d.resolution_time_ms for d in self.resolution_history) / total,
        }
