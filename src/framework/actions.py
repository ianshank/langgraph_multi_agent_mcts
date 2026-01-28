"""
MCTS Action Definitions.

Defines concrete actions available to the MCTS engine for graph traversal.
Implements the Command pattern for reasoning steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from .mcts.core import MCTSState


@dataclass
class Action(ABC):
    """Base class for MCTS actions."""

    name: str
    cost: float = 1.0

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the type of action."""
        pass

    def apply(self, state: MCTSState) -> MCTSState:
        """
        Apply this action to a state to produce a new state.
        
        Args:
            state: Current MCTSState
            
        Returns:
            New MCTSState
        """
        # Create new ID based on action
        new_id = f"{state.state_id}_{self.name}"
        
        # Copy features
        new_features = state.features.copy()
        
        # Update history
        history = new_features.get("history", [])
        new_features["history"] = history + [self.name]
        
        # Update specific logic
        self._update_features(new_features)
        
        return MCTSState(
            state_id=new_id,
            features=new_features
        )

    def _update_features(self, features: dict[str, Any]) -> None:
        """Hook for subclasses to update state features."""
        pass


@dataclass
class DecomposeAction(Action):
    """Action to break a complex query into sub-problems."""
    
    type: str = "decompose"
    
    def _update_features(self, features: dict[str, Any]) -> None:
        features["depth"] = features.get("depth", 0) + 1
        features["is_decomposed"] = True


@dataclass
class ResearchAction(Action):
    """Action to perform retrieval or external search."""
    
    topic: str = ""
    type: str = "research"
    
    def _update_features(self, features: dict[str, Any]) -> None:
        features["has_context"] = True
        # Simulate gaining information reduces uncertainty
        features["uncertainty"] = features.get("uncertainty", 1.0) * 0.5


@dataclass
class SynthesizeAction(Action):
    """Action to combine findings into an answer."""
    
    type: str = "synthesize"
    
    def _update_features(self, features: dict[str, Any]) -> None:
        features["is_terminal"] = True
        features["complete"] = True


@dataclass
class CritiqueAction(Action):
    """Action to review and critique current findings."""
    
    type: str = "critique"
    
    def _update_features(self, features: dict[str, Any]) -> None:
        features["quality_check"] = True
        # Critique might increase depth/steps needed if issues found
        # but we'll model it as a refinement step here.


class ActionRegistry:
    """Registry of available actions for the MCTS engine."""
    
    COMMON_ACTIONS: ClassVar[list[Action]] = [
        DecomposeAction(name="decompose_problem"),
        ResearchAction(name="research_context"),
        SynthesizeAction(name="draft_answer"),
        CritiqueAction(name="review_logic"),
    ]
    
    @classmethod
    def get_available_actions(cls, state: MCTSState) -> list[str]:
        """
        Get valid action names for a given state.
        
        Args:
            state: Current MCTS state
            
        Returns:
            List of action name strings
        """
        features = state.features
        history = features.get("history", [])
        
        # Terminal state check
        if features.get("is_terminal") or features.get("complete"):
            return []
            
        # Logic for available actions based on state
        available = []
        
        # Can always decompose if not too deep
        depth = features.get("depth", 0)
        if depth < 3 and not features.get("is_decomposed"):
            available.append("decompose_problem")
            
        # Can research if not already done heavily
        research_count = sum(1 for a in history if "research" in a)
        if research_count < 3:
            available.append("research_context")
            
        # Can synthesize if we have context or decomposition
        if features.get("has_context") or features.get("is_decomposed"):
            available.append("draft_answer")
            
        # Can critique if we have drafted something (not strictly enforced in simplified model)
        if "draft_answer" in history and "review_logic" not in history:
            available.append("review_logic")
            
        # Fallback if list empty but not terminal (to avoid dead ends in simulation)
        if not available and not features.get("complete"):
             available.append("draft_answer")
             
        return available

    @classmethod
    def get_action_by_name(cls, name: str) -> Action | None:
        """Retrieve Action instance by name."""
        for action in cls.COMMON_ACTIONS:
            if action.name == name:
                return action
        return None

