"""
Semantic Consensus Framework (SCF) - Main middleware class.

Intercepts agent actions, analyzes for conflicts, resolves them,
and monitors for drift. This is the primary API surface.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from scf.core.sig import SemanticIntentGraph, IntentNode, IntentStatus
from scf.core.pcl import ProcessContextLayer
from scf.detect.cde import ConflictDetectionEngine, DetectedConflict
from scf.resolve.crp import ConsensusResolutionProtocol, ResolutionDecision, PolicyRule, ResolutionOutcome
from scf.drift.monitor import DriftMonitor, AgentState
from scf.governance.pagi import GovernanceIntegration


@dataclass
class SCFConfig:
    """Configuration for the SCF middleware."""
    use_embeddings: bool = False  # Disable by default for speed; enable for production
    drift_threshold: float = 0.6
    drift_check_interval: int = 3  # Check drift every N interactions
    enable_governance: bool = True
    enable_drift_monitor: bool = True


@dataclass
class InteractionResult:
    """Result of processing an agent action through SCF."""
    intent_id: str
    approved: bool
    conflicts_detected: List[DetectedConflict] = field(default_factory=list)
    resolutions: List[ResolutionDecision] = field(default_factory=list)
    drift_events: List = field(default_factory=list)
    processing_time_ms: float = 0.0
    blocked_reason: Optional[str] = None


class SemanticConsensusFramework:
    """
    Main SCF middleware. Drop-in layer between orchestrator and agent execution.
    
    Usage:
        pcl = ProcessContextLayer()
        pcl.load_from_yaml("process_model.yaml")
        scf = SemanticConsensusFramework(pcl)
        
        # Before agent executes:
        result = scf.process_intent(intent_node)
        if result.approved:
            # Execute the action
            scf.mark_executed(intent_node.id)
    """

    def __init__(self, pcl: ProcessContextLayer, config: SCFConfig = None):
        self.config = config or SCFConfig()
        self.pcl = pcl
        self.sig = SemanticIntentGraph()
        self.cde = ConflictDetectionEngine(pcl, use_embeddings=self.config.use_embeddings)
        self.crp = ConsensusResolutionProtocol(pcl)
        self.drift_monitor = DriftMonitor(pcl, threshold=self.config.drift_threshold) if self.config.enable_drift_monitor else None
        self.governance = GovernanceIntegration() if self.config.enable_governance else None
        self._interaction_count = 0
        self._total_overhead_ms = 0.0
        self._interaction_times: List[float] = []

    def add_policy_rule(self, rule: PolicyRule):
        """Add a governance policy rule for conflict resolution."""
        self.crp.add_policy_rule(rule)

    def set_role_authority(self, role: str, level: int):
        """Set authority level for an agent role."""
        self.crp.set_role_authority(role, level)

    def process_intent(self, intent: IntentNode) -> InteractionResult:
        """
        Process an agent's intent through the full SCF pipeline.
        This is the main API method called before any agent action.
        """
        start = time.time()
        self._interaction_count += 1

        # Step 1: Register intent in SIG
        self.sig.register_intent(intent)

        # Step 2: Detect conflicts
        conflicts = self.cde.detect_conflicts(self.sig)

        # Filter to conflicts involving this intent
        relevant_conflicts = [
            c for c in conflicts
            if c.intent_a_id == intent.id or c.intent_b_id == intent.id
        ]

        # Step 3: Resolve conflicts
        resolutions = []
        blocked = False
        blocked_reason = None

        for conflict in relevant_conflicts:
            if self.governance:
                self.governance.log_conflict(conflict)

            decision = self.crp.resolve(conflict, self.sig)
            resolutions.append(decision)

            if self.governance:
                self.governance.log_resolution(decision)

            # Check if this intent was blocked
            if decision.loser_intent_id == intent.id:
                blocked = True
                blocked_reason = decision.reason
            elif decision.outcome == ResolutionOutcome.ESCALATED:
                blocked = True
                blocked_reason = "Escalated to human review"

        # Step 4: If no conflicts, approve
        if not relevant_conflicts:
            self.sig.update_status(intent.id, IntentStatus.APPROVED)

        # Step 5: Drift monitoring (periodic)
        drift_events = []
        if self.drift_monitor and self._interaction_count % self.config.drift_check_interval == 0:
            drift_events = self.drift_monitor.check_alignment(self.sig, self._interaction_count)
            if self.governance:
                for de in drift_events:
                    self.governance.log_drift(de)

        elapsed = (time.time() - start) * 1000
        self._total_overhead_ms += elapsed
        self._interaction_times.append(elapsed)

        return InteractionResult(
            intent_id=intent.id,
            approved=not blocked,
            conflicts_detected=relevant_conflicts,
            resolutions=resolutions,
            drift_events=drift_events,
            processing_time_ms=elapsed,
            blocked_reason=blocked_reason,
        )

    def update_agent_state(self, state: AgentState):
        """Update drift monitor with agent's current state."""
        if self.drift_monitor:
            self.drift_monitor.update_agent_state(state)

    def mark_executed(self, intent_id: str):
        """Mark an intent as executed after successful action."""
        self.sig.update_status(intent_id, IntentStatus.EXECUTED)

    def get_stats(self) -> Dict:
        """Return comprehensive SCF statistics."""
        latencies = sorted(self._interaction_times)
        n = len(latencies)

        stats = {
            "total_interactions": self._interaction_count,
            "sig": self.sig.get_stats(),
            "detection": self.cde.get_detection_stats(),
            "resolution": self.crp.get_resolution_stats(),
        }

        if n > 0:
            stats["latency"] = {
                "median_ms": latencies[n // 2] if n > 0 else 0,
                "p95_ms": latencies[int(n * 0.95)] if n > 0 else 0,
                "mean_ms": sum(latencies) / n,
                "total_overhead_ms": self._total_overhead_ms,
            }

        if self.drift_monitor:
            stats["drift"] = self.drift_monitor.get_drift_stats()

        if self.governance:
            stats["governance"] = self.governance.get_dashboard_data()

        return stats

    def reset(self):
        """Reset SCF state for a new workflow run."""
        self.sig = SemanticIntentGraph()
        self.cde = ConflictDetectionEngine(self.pcl, use_embeddings=self.config.use_embeddings)
        self.crp = ConsensusResolutionProtocol(self.pcl)
        # Re-add policy rules and authorities
        if self.drift_monitor:
            self.drift_monitor = DriftMonitor(self.pcl, threshold=self.config.drift_threshold)
        if self.governance:
            self.governance = GovernanceIntegration()
        self._interaction_count = 0
        self._total_overhead_ms = 0.0
        self._interaction_times = []
