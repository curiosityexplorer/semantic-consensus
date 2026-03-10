"""
Process Context Layer (PCL) - Shared operational semantic foundation.

Ingests enterprise workflow models and establishes a shared vocabulary
that all agents reference for consistent intent interpretation.
"""

import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path


@dataclass
class StateTransition:
    """A valid state transition in the process model."""
    from_state: str
    to_state: str
    required_role: Optional[str] = None  # Role authorized to make this transition
    conditions: List[str] = field(default_factory=list)


@dataclass
class EntityModel:
    """Model for a business entity with valid states and transitions."""
    entity_type: str
    valid_states: List[str]
    terminal_states: List[str] = field(default_factory=list)
    mutually_exclusive_states: List[Tuple[str, str]] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    state_synonyms: Dict[str, str] = field(default_factory=dict)  # synonym -> canonical

    def are_mutually_exclusive(self, state_a: str, state_b: str) -> bool:
        """Check if two states are mutually exclusive."""
        a = self.resolve_synonym(state_a)
        b = self.resolve_synonym(state_b)
        return (a, b) in self.mutually_exclusive_states or (b, a) in self.mutually_exclusive_states

    def resolve_synonym(self, state: str) -> str:
        """Resolve a state synonym to its canonical form."""
        return self.state_synonyms.get(state.lower(), state.lower())

    def is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a state transition is valid."""
        f = self.resolve_synonym(from_state)
        t = self.resolve_synonym(to_state)
        return any(
            tr.from_state == f and tr.to_state == t
            for tr in self.transitions
        )

    def is_terminal(self, state: str) -> bool:
        """Check if a state is terminal."""
        return self.resolve_synonym(state) in self.terminal_states


@dataclass
class AuthorityMapping:
    """Maps roles to their decision authority over entity state transitions."""
    role: str
    entity_type: str
    allowed_transitions: List[Tuple[str, str]]  # (from_state, to_state)
    priority: int = 0  # Higher = more authority


@dataclass
class TemporalConstraint:
    """Defines required ordering between operations."""
    operation_a: str  # Must happen before
    operation_b: str  # Must happen after
    entity_type: str
    description: str = ""


class ProcessSemanticModel:
    """
    Unified process semantic model that the PCL generates from enterprise workflow inputs.
    Provides shared ground truth for all agents.
    """

    def __init__(self):
        self.entity_models: Dict[str, EntityModel] = {}
        self.authority_mappings: List[AuthorityMapping] = []
        self.temporal_constraints: List[TemporalConstraint] = []
        self.shared_vocabulary: Dict[str, str] = {}  # term -> canonical meaning
        self.resource_constraints: Dict[str, Dict] = {}  # entity -> {field: constraint}

    def add_entity_model(self, model: EntityModel):
        self.entity_models[model.entity_type] = model

    def add_authority(self, mapping: AuthorityMapping):
        self.authority_mappings.append(mapping)

    def add_temporal_constraint(self, constraint: TemporalConstraint):
        self.temporal_constraints.append(constraint)

    def add_resource_constraint(self, entity_type: str, field: str, constraint: Dict):
        if entity_type not in self.resource_constraints:
            self.resource_constraints[entity_type] = {}
        self.resource_constraints[entity_type][field] = constraint

    def check_state_exclusivity(self, entity_type: str, state_a: str, state_b: str) -> bool:
        """Check if two states are mutually exclusive for an entity type."""
        model = self.entity_models.get(entity_type)
        if not model:
            return False
        return model.are_mutually_exclusive(state_a, state_b)

    def check_valid_transition(self, entity_type: str, from_state: str, to_state: str) -> bool:
        """Check if a state transition is valid."""
        model = self.entity_models.get(entity_type)
        if not model:
            return True  # Unknown entity types are permissive
        return model.is_valid_transition(from_state, to_state)

    def get_role_authority(self, role: str, entity_type: str) -> int:
        """Get the authority level for a role over an entity type."""
        for mapping in self.authority_mappings:
            if mapping.role == role and mapping.entity_type == entity_type:
                return mapping.priority
        return 0

    def check_temporal_ordering(self, op_a: str, op_b: str, entity_type: str) -> Optional[bool]:
        """
        Check if op_a must happen before op_b.
        Returns True if ordering required, False if reverse required, None if no constraint.
        """
        for tc in self.temporal_constraints:
            if tc.entity_type == entity_type:
                if tc.operation_a == op_a and tc.operation_b == op_b:
                    return True
                if tc.operation_a == op_b and tc.operation_b == op_a:
                    return False
        return None

    def resolve_term(self, term: str) -> str:
        """Resolve a term to its canonical form using the shared vocabulary."""
        return self.shared_vocabulary.get(term.lower(), term.lower())

    def check_resource_constraint(self, entity_type: str, field: str, value) -> bool:
        """Check if a value satisfies a resource constraint."""
        constraints = self.resource_constraints.get(entity_type, {}).get(field)
        if not constraints:
            return True
        if "min" in constraints and value < constraints["min"]:
            return False
        if "max" in constraints and value > constraints["max"]:
            return False
        return True


class ProcessContextLayer:
    """
    Main PCL class that ingests process models and provides
    scoped context injection for agents.
    """

    def __init__(self):
        self.psm = ProcessSemanticModel()

    def load_from_yaml(self, path: str) -> 'ProcessContextLayer':
        """Load a process model from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self._parse_model(data)
        return self

    def load_from_dict(self, data: Dict) -> 'ProcessContextLayer':
        """Load a process model from a dictionary."""
        self._parse_model(data)
        return self

    def _parse_model(self, data: Dict):
        """Parse a process model definition into the PSM."""
        # Parse entity models
        for entity_data in data.get("entities", []):
            model = EntityModel(
                entity_type=entity_data["type"],
                valid_states=entity_data.get("states", []),
                terminal_states=entity_data.get("terminal_states", []),
                mutually_exclusive_states=[
                    tuple(pair) for pair in entity_data.get("mutually_exclusive", [])
                ],
                transitions=[
                    StateTransition(
                        from_state=t["from"],
                        to_state=t["to"],
                        required_role=t.get("role"),
                        conditions=t.get("conditions", [])
                    )
                    for t in entity_data.get("transitions", [])
                ],
                state_synonyms=entity_data.get("synonyms", {})
            )
            self.psm.add_entity_model(model)

        # Parse authority mappings
        for auth_data in data.get("authorities", []):
            self.psm.add_authority(AuthorityMapping(
                role=auth_data["role"],
                entity_type=auth_data["entity_type"],
                allowed_transitions=[
                    tuple(t) for t in auth_data.get("transitions", [])
                ],
                priority=auth_data.get("priority", 0)
            ))

        # Parse temporal constraints
        for tc_data in data.get("temporal_constraints", []):
            self.psm.add_temporal_constraint(TemporalConstraint(
                operation_a=tc_data["before"],
                operation_b=tc_data["after"],
                entity_type=tc_data["entity_type"],
                description=tc_data.get("description", "")
            ))

        # Parse shared vocabulary
        self.psm.shared_vocabulary.update(data.get("vocabulary", {}))

        # Parse resource constraints
        for rc_data in data.get("resource_constraints", []):
            self.psm.add_resource_constraint(
                entity_type=rc_data["entity_type"],
                field=rc_data["field"],
                constraint=rc_data["constraint"]
            )

    def get_agent_context(self, agent_role: str, entity_types: List[str]) -> Dict:
        """
        Generate scoped context for an agent based on its role and relevant entities.
        This is what gets injected into the agent's context window.
        """
        context = {
            "role": agent_role,
            "entity_models": {},
            "authority": [],
            "temporal_constraints": [],
            "vocabulary": self.psm.shared_vocabulary,
        }

        for et in entity_types:
            if et in self.psm.entity_models:
                model = self.psm.entity_models[et]
                context["entity_models"][et] = {
                    "valid_states": model.valid_states,
                    "terminal_states": model.terminal_states,
                    "mutually_exclusive": model.mutually_exclusive_states,
                    "synonyms": model.state_synonyms,
                }

        for auth in self.psm.authority_mappings:
            if auth.role == agent_role:
                context["authority"].append({
                    "entity_type": auth.entity_type,
                    "transitions": auth.allowed_transitions,
                    "priority": auth.priority,
                })

        for tc in self.psm.temporal_constraints:
            if tc.entity_type in entity_types:
                context["temporal_constraints"].append({
                    "before": tc.operation_a,
                    "after": tc.operation_b,
                    "entity_type": tc.entity_type,
                })

        return context
