"""
Feature Extractor for Meta-Controller - SAFE VERSION.

This version uses ONLY heuristic-based feature extraction to avoid
any dependency issues with sentence-transformers.
"""

import logging
import os
from dataclasses import dataclass

from src.agents.meta_controller.base import MetaControllerFeatures

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractorConfig:
    """Configuration for FeatureExtractor."""

    model_name: str = "heuristic-only"
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "FeatureExtractorConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "heuristic-only"),
            device=os.getenv("DEVICE", "cpu"),
        )


class FeatureExtractor:
    """
    Extracts features from queries using heuristic analysis only.

    This safe version does not use any ML dependencies to ensure
    compatibility across all environments.
    """

    # Agent keywords for heuristic matching
    AGENT_KEYWORDS = {
        "hrm": [
            "complex",
            "decompose",
            "break",
            "hierarchical",
            "structure",
            "multiple",
            "questions",
            "planning",
            "analyze",
            "components",
        ],
        "trm": ["iterate", "refine", "improve", "compare", "versus", "vs", "fix", "polish", "enhance", "better"],
        "mcts": [
            "optimize",
            "best",
            "strategic",
            "search",
            "explore",
            "path",
            "decision",
            "uncertainty",
            "maximize",
            "performance",
        ],
    }

    def __init__(self, config: FeatureExtractorConfig | None = None):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration object
        """
        if config is None:
            config = FeatureExtractorConfig()

        self.config = config
        logger.info(f"FeatureExtractor initialized with heuristic mode (no ML dependencies)")

    def extract_features(self, query: str, iteration: int = 0, last_agent: str = "none") -> MetaControllerFeatures:
        """
        Extract features from a query using heuristic analysis.

        Args:
            query: The input query text
            iteration: Current iteration number
            last_agent: Name of the last agent used

        Returns:
            MetaControllerFeatures object populated with heuristic scores
        """
        query_lower = query.lower()

        # Count keyword matches for each agent
        keyword_scores = {}
        for agent, keywords in self.AGENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            # Normalize by number of keywords to get a ratio
            keyword_scores[agent] = score / len(keywords) if keywords else 0

        # Additional heuristics
        has_multiple_questions = "?" in query and query.count("?") > 1
        has_comparison = any(word in query_lower for word in ["vs", "versus", "compare", "difference", "between"])
        has_optimization = any(word in query_lower for word in ["optimize", "best", "improve", "maximize", "enhance"])
        has_technical = any(word in query_lower for word in ["algorithm", "code", "implement", "technical", "system"])
        query_complexity = min(len(query) / 500, 1.0)  # Normalize by typical query length

        # Calculate confidence scores with keyword matches and heuristics
        hrm_confidence = (
            keyword_scores["hrm"] * 0.5
            + (0.2 if has_multiple_questions else 0)
            + (0.15 if has_technical else 0)
            + (0.15 * query_complexity)
        )

        trm_confidence = (
            keyword_scores["trm"] * 0.5
            + (0.25 if has_comparison else 0)
            + (0.15 if len(query) > 100 else 0)
            + (0.1 if "refine" in query_lower or "improve" in query_lower else 0)
        )

        mcts_confidence = (
            keyword_scores["mcts"] * 0.5
            + (0.25 if has_optimization else 0)
            + (0.15 if has_technical else 0)
            + (0.1 if "search" in query_lower or "explore" in query_lower else 0)
        )

        # Add base confidence to avoid all zeros
        hrm_confidence = max(0.1, min(1.0, hrm_confidence + 0.2))
        trm_confidence = max(0.1, min(1.0, trm_confidence + 0.2))
        mcts_confidence = max(0.1, min(1.0, mcts_confidence + 0.2))

        # Normalize to sum to approximately 1
        total = hrm_confidence + trm_confidence + mcts_confidence
        if total > 0:
            hrm_confidence /= total
            trm_confidence /= total
            mcts_confidence /= total
        else:
            # Equal distribution if no signals
            hrm_confidence = trm_confidence = mcts_confidence = 1.0 / 3.0

        # Calculate consensus score (how much agents agree)
        confidences = [hrm_confidence, trm_confidence, mcts_confidence]
        max_conf = max(confidences)
        min_conf = min(confidences)
        consensus_score = min_conf / max_conf if max_conf > 0 else 0.0

        # Penalize if we're stuck with the same agent
        if last_agent in ["hrm", "trm", "mcts"]:
            if iteration > 2:
                # Reduce confidence for the last agent to encourage switching
                if last_agent == "hrm":
                    hrm_confidence *= 0.7
                elif last_agent == "trm":
                    trm_confidence *= 0.7
                elif last_agent == "mcts":
                    mcts_confidence *= 0.7

                # Renormalize
                total = hrm_confidence + trm_confidence + mcts_confidence
                if total > 0:
                    hrm_confidence /= total
                    trm_confidence /= total
                    mcts_confidence /= total

        return MetaControllerFeatures(
            hrm_confidence=float(hrm_confidence),
            trm_confidence=float(trm_confidence),
            mcts_value=float(mcts_confidence),
            consensus_score=float(consensus_score),
            last_agent=last_agent,
            iteration=iteration,
            query_length=len(query),
            has_rag_context=len(query) > 500,
            rag_relevance_score=0.0,  # Not available without embeddings
            is_technical_query=has_technical,
        )

    def __repr__(self) -> str:
        """String representation of the feature extractor."""
        return f"FeatureExtractor(mode=heuristic-only, config={self.config})"
