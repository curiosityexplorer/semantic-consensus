"""
Process-Aware Governance Integration (PAGI) - Connects SCF to enterprise governance.

Provides: Policy Mapping, Audit Trail Generation, Governance Dashboard data.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from scf.detect.cde import DetectedConflict
from scf.resolve.crp import ResolutionDecision
from scf.drift.monitor import DriftEvent


@dataclass
class AuditEntry:
    """Single entry in the governance audit trail."""
    timestamp: str
    event_type: str  # "conflict_detected", "conflict_resolved", "drift_detected", "resync"
    details: Dict
    agents_involved: List[str]
    resolution_tier: Optional[str] = None
    policy_applied: Optional[str] = None
    outcome: Optional[str] = None


class GovernanceIntegration:
    """
    Bridges SCF with enterprise governance infrastructure.
    Maintains complete audit trail and governance metrics.
    """

    def __init__(self):
        self.audit_trail: List[AuditEntry] = []
        self._governance_policies: Dict[str, Dict] = {}
        self._metrics = {
            "total_conflicts": 0,
            "total_resolutions": 0,
            "total_drift_events": 0,
            "total_escalations": 0,
            "policy_coverage_hits": 0,
            "policy_coverage_misses": 0,
        }

    def log_conflict(self, conflict: DetectedConflict):
        """Log a detected conflict to the audit trail."""
        self._metrics["total_conflicts"] += 1
        self.audit_trail.append(AuditEntry(
            timestamp=datetime.now().isoformat(),
            event_type="conflict_detected",
            details={
                "conflict_id": conflict.conflict_id,
                "conflict_type": conflict.conflict_type.value,
                "severity": conflict.severity.value,
                "entity": conflict.entity,
                "description": conflict.description,
                "confidence": conflict.confidence,
                "detection_method": conflict.detection_method,
                "detection_time_ms": conflict.detection_time_ms,
            },
            agents_involved=[conflict.agent_a_id, conflict.agent_b_id],
        ))

    def log_resolution(self, decision: ResolutionDecision):
        """Log a conflict resolution to the audit trail."""
        self._metrics["total_resolutions"] += 1
        if decision.resolution_tier.value == "escalation":
            self._metrics["total_escalations"] += 1
        if decision.policy_applied:
            self._metrics["policy_coverage_hits"] += 1
        else:
            self._metrics["policy_coverage_misses"] += 1

        self.audit_trail.append(AuditEntry(
            timestamp=datetime.now().isoformat(),
            event_type="conflict_resolved",
            details={
                "conflict_id": decision.conflict_id,
                "outcome": decision.outcome.value,
                "winner_intent": decision.winner_intent_id,
                "loser_intent": decision.loser_intent_id,
                "reason": decision.reason,
                "resolution_time_ms": decision.resolution_time_ms,
            },
            agents_involved=[],
            resolution_tier=decision.resolution_tier.value,
            policy_applied=decision.policy_applied,
            outcome=decision.outcome.value,
        ))

    def log_drift(self, event: DriftEvent):
        """Log a drift event to the audit trail."""
        self._metrics["total_drift_events"] += 1
        self.audit_trail.append(AuditEntry(
            timestamp=datetime.now().isoformat(),
            event_type="drift_detected",
            details={
                "event_id": event.event_id,
                "sas_score": event.sas_score,
                "threshold": event.threshold,
                "entities_diverged": event.entities_diverged,
                "interaction_step": event.interaction_step,
                "resync_triggered": event.resync_triggered,
            },
            agents_involved=[event.agent_a_id, event.agent_b_id],
        ))

    def get_dashboard_data(self) -> Dict:
        """Return data for the governance dashboard."""
        return {
            "metrics": self._metrics.copy(),
            "policy_coverage_rate": (
                self._metrics["policy_coverage_hits"] /
                max(self._metrics["total_resolutions"], 1) * 100
            ),
            "escalation_rate": (
                self._metrics["total_escalations"] /
                max(self._metrics["total_resolutions"], 1) * 100
            ),
            "audit_entries": len(self.audit_trail),
        }

    def export_audit_trail(self, filepath: str):
        """Export audit trail to JSON."""
        data = [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "details": e.details,
                "agents_involved": e.agents_involved,
                "resolution_tier": e.resolution_tier,
                "policy_applied": e.policy_applied,
                "outcome": e.outcome,
            }
            for e in self.audit_trail
        ]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
