"""
Clinical Trial Design Optimizer Use Case.

Uses MCTS to simulate millions of trial designs, patient cohort strategies,
and endpoint combinations to maximize approval probability while minimizing
cost and timeline.

Target Buyers: Top 20 pharma companies (Pfizer, Roche, J&J)
Revenue Potential: $10M-50M ARR
"""

from __future__ import annotations

from .use_case import ClinicalTrialDesign

__all__ = ["ClinicalTrialDesign"]
