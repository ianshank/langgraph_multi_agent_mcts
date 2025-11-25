"""
Google ADK (Agent Development Kit) integration for LangGraph multi-agent framework.

This module provides adapters and wrappers for integrating Google ADK agents
into the LangGraph-based MCTS framework, enabling hybrid workflows that combine
specialized Google agents with custom HRM/TRM agents.

Available Agents:
- MLEngineeringAgent: Machine learning engineering and model training
- DataScienceAgent: Data analysis, NL2SQL, and BigQuery integration
- AcademicResearchAgent: Research paper analysis and citation discovery
- DataEngineeringAgent: Dataform pipeline development and management
- DeepSearchAgent: Production-ready research with human-in-the-loop
"""

from .agents.academic_research import AcademicResearchAgent
from .agents.data_engineering import DataEngineeringAgent
from .agents.data_science import DataScienceAgent
from .agents.deep_search import DeepSearchAgent
from .agents.ml_engineering import MLEngineeringAgent
from .base import ADKAgentAdapter, ADKConfig

__all__ = [
    "ADKAgentAdapter",
    "ADKConfig",
    "MLEngineeringAgent",
    "DataScienceAgent",
    "AcademicResearchAgent",
    "DataEngineeringAgent",
    "DeepSearchAgent",
]

__version__ = "0.1.0"
