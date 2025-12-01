"""
Streamlit UI for LangGraph Multi-Agent MCTS Framework.

Provides an interactive interface for:
- Query processing with configurable agents
- MCTS visualization and configuration
- Expert Iteration training dashboard
- Real-time metrics and monitoring
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# Import framework components
from src.framework.actions import (
    ActionType,
    AgentType,
    GraphConfig,
    ConfidenceConfig,
    RolloutWeights,
    SynthesisConfig,
    create_research_config,
    create_coding_config,
    create_creative_config,
)
from src.framework.mcts.config import (
    MCTSConfig,
    ConfigPreset,
    create_preset_config,
)

# Conditional imports for optional features
try:
    from src.framework.mcts.neural_integration import (
        NeuralMCTSConfig,
        get_fast_neural_config,
        get_balanced_neural_config,
        get_thorough_neural_config,
        get_alphazero_config,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

try:
    from src.training.expert_iteration import ExpertIterationConfig
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="LangGraph Multi-Agent MCTS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "query_history": [],
        "current_result": None,
        "mcts_stats": None,
        "training_metrics": [],
        "config_preset": "balanced",
        "domain_preset": "general",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("Configuration")

    # Domain preset selection
    st.sidebar.subheader("Domain Preset")
    domain = st.sidebar.selectbox(
        "Select Domain",
        ["General", "Research", "Coding", "Creative", "Custom"],
        key="domain_select",
    )

    if domain == "Custom":
        render_custom_config()
    else:
        st.session_state.domain_preset = domain.lower()

    st.sidebar.divider()

    # MCTS Configuration
    st.sidebar.subheader("MCTS Settings")
    mcts_preset = st.sidebar.selectbox(
        "MCTS Preset",
        ["Fast", "Balanced", "Thorough", "Exploration Heavy", "Exploitation Heavy"],
        index=1,
        key="mcts_preset",
    )

    # Show MCTS details
    with st.sidebar.expander("MCTS Details"):
        preset_map = {
            "Fast": ConfigPreset.FAST,
            "Balanced": ConfigPreset.BALANCED,
            "Thorough": ConfigPreset.THOROUGH,
            "Exploration Heavy": ConfigPreset.EXPLORATION_HEAVY,
            "Exploitation Heavy": ConfigPreset.EXPLOITATION_HEAVY,
        }
        config = create_preset_config(preset_map[mcts_preset])
        st.json({
            "iterations": config.num_iterations,
            "exploration_weight": config.exploration_weight,
            "max_rollout_depth": config.max_rollout_depth,
            "progressive_widening_k": config.progressive_widening_k,
        })

    st.sidebar.divider()

    # Neural MCTS toggle
    if NEURAL_AVAILABLE:
        st.sidebar.subheader("Neural MCTS")
        use_neural = st.sidebar.checkbox("Enable Neural MCTS", value=False)
        if use_neural:
            neural_preset = st.sidebar.selectbox(
                "Neural Preset",
                ["Fast", "Balanced", "Thorough", "AlphaZero"],
                index=1,
            )
            st.session_state.neural_preset = neural_preset.lower()

    st.sidebar.divider()

    # Agent toggles
    st.sidebar.subheader("Agents")
    st.session_state.use_hrm = st.sidebar.checkbox("HRM (Hierarchical)", value=True)
    st.session_state.use_trm = st.sidebar.checkbox("TRM (Recursive)", value=True)
    st.session_state.use_mcts = st.sidebar.checkbox("MCTS Search", value=True)
    st.session_state.use_parallel = st.sidebar.checkbox("Parallel Execution", value=True)


def render_custom_config():
    """Render custom configuration options."""
    with st.sidebar.expander("Custom Configuration", expanded=True):
        # Confidence settings
        st.write("**Confidence Settings**")
        consensus_threshold = st.slider(
            "Consensus Threshold",
            0.5, 1.0, 0.75, 0.05,
            help="Minimum agreement required between agents"
        )

        # Rollout weights
        st.write("**Rollout Weights**")
        heuristic_weight = st.slider(
            "Heuristic Weight",
            0.0, 1.0, 0.7, 0.1,
        )

        # Synthesis settings
        st.write("**Synthesis Settings**")
        temperature = st.slider(
            "Temperature",
            0.0, 2.0, 0.5, 0.1,
        )

        # Store custom config
        st.session_state.custom_config = {
            "consensus_threshold": consensus_threshold,
            "heuristic_weight": heuristic_weight,
            "random_weight": 1.0 - heuristic_weight,
            "temperature": temperature,
        }


def render_main_query_interface():
    """Render the main query interface."""
    st.header("Multi-Agent Reasoning")

    # Query input
    query = st.text_area(
        "Enter your query",
        placeholder="Ask a complex question that benefits from multi-agent reasoning...",
        height=100,
        key="query_input",
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        use_rag = st.checkbox("Enable RAG", value=False)

    with col2:
        max_iterations = st.number_input("Max Iterations", 1, 10, 3)

    with col3:
        submit = st.button("Process Query", type="primary", use_container_width=True)

    if submit and query:
        process_query(query, use_rag, max_iterations)

    # Display results
    if st.session_state.current_result:
        render_results()


def process_query(query: str, use_rag: bool, max_iterations: int):
    """Process a query through the multi-agent system."""
    with st.spinner("Processing with multi-agent reasoning..."):
        # Simulate processing (replace with actual implementation)
        start_time = time.time()

        # Create progress indicators
        progress = st.progress(0)
        status = st.empty()

        # Simulate agent execution
        agents_executed = []

        if st.session_state.get("use_hrm", True):
            status.text("Running HRM agent...")
            progress.progress(25)
            time.sleep(0.5)
            agents_executed.append({
                "agent": "HRM",
                "confidence": 0.85,
                "response": "Decomposed query into 3 sub-problems",
            })

        if st.session_state.get("use_trm", True):
            status.text("Running TRM agent...")
            progress.progress(50)
            time.sleep(0.5)
            agents_executed.append({
                "agent": "TRM",
                "confidence": 0.82,
                "response": "Refined solution through 4 iterations",
            })

        if st.session_state.get("use_mcts", True):
            status.text("Running MCTS search...")
            progress.progress(75)
            time.sleep(0.5)
            agents_executed.append({
                "agent": "MCTS",
                "confidence": 0.88,
                "response": "Explored 100 decision paths",
            })

        status.text("Synthesizing response...")
        progress.progress(100)

        elapsed = time.time() - start_time

        # Store result
        st.session_state.current_result = {
            "query": query,
            "response": f"Based on multi-agent analysis of '{query[:50]}...', here is the synthesized response combining insights from {len(agents_executed)} agents.",
            "agents": agents_executed,
            "consensus_score": 0.85,
            "elapsed_time": elapsed,
            "mcts_stats": {
                "iterations": 100,
                "best_action": "synthesize",
                "best_action_visits": 35,
                "best_action_value": 0.87,
                "cache_hit_rate": 0.23,
            },
        }

        # Add to history
        st.session_state.query_history.append({
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "consensus_score": 0.85,
        })

        progress.empty()
        status.empty()


def render_results():
    """Render the query results."""
    result = st.session_state.current_result

    st.divider()

    # Response
    st.subheader("Response")
    st.success(result["response"])

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Consensus Score", f"{result['consensus_score']:.2%}")

    with col2:
        st.metric("Agents Used", len(result["agents"]))

    with col3:
        st.metric("Processing Time", f"{result['elapsed_time']:.2f}s")

    with col4:
        if result.get("mcts_stats"):
            st.metric("MCTS Iterations", result["mcts_stats"]["iterations"])

    # Agent details
    st.subheader("Agent Outputs")

    for agent in result["agents"]:
        with st.expander(f"{agent['agent']} Agent (Confidence: {agent['confidence']:.2%})"):
            st.write(agent["response"])

    # MCTS visualization
    if result.get("mcts_stats"):
        st.subheader("MCTS Analysis")
        render_mcts_visualization(result["mcts_stats"])


def render_mcts_visualization(stats: dict):
    """Render MCTS statistics visualization."""
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Search Statistics**")
        st.json(stats)

    with col2:
        st.write("**Action Distribution**")
        # Simulated action distribution
        actions = {
            "explore_breadth": 25,
            "explore_depth": 20,
            "synthesize": 35,
            "delegate": 20,
        }

        import pandas as pd
        df = pd.DataFrame({
            "Action": list(actions.keys()),
            "Visits": list(actions.values()),
        })
        st.bar_chart(df.set_index("Action"))


def render_training_dashboard():
    """Render the Expert Iteration training dashboard."""
    st.header("Expert Iteration Training")

    if not TRAINING_AVAILABLE:
        st.warning("Training module not available. Install PyTorch to enable.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Configuration")

        episodes = st.number_input("Episodes per Iteration", 10, 1000, 100)
        simulations = st.number_input("MCTS Simulations", 50, 3200, 400)
        batch_size = st.number_input("Batch Size", 32, 2048, 256)
        learning_rate = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f")

        if st.button("Start Training", type="primary"):
            run_training_simulation(episodes, simulations, batch_size, learning_rate)

    with col2:
        st.subheader("Training Status")

        if st.session_state.training_metrics:
            latest = st.session_state.training_metrics[-1]
            st.metric("Current Iteration", latest["iteration"])
            st.metric("Average Outcome", f"{latest['avg_outcome']:.3f}")
            st.metric("Buffer Size", latest["buffer_size"])
        else:
            st.info("No training in progress")

    # Training history
    if st.session_state.training_metrics:
        st.subheader("Training Progress")

        import pandas as pd
        df = pd.DataFrame(st.session_state.training_metrics)

        col1, col2 = st.columns(2)

        with col1:
            st.line_chart(df[["iteration", "avg_outcome"]].set_index("iteration"))

        with col2:
            if "policy_loss" in df.columns:
                st.line_chart(df[["iteration", "policy_loss", "value_loss"]].set_index("iteration"))


def run_training_simulation(episodes: int, simulations: int, batch_size: int, lr: float):
    """Simulate training progress."""
    progress = st.progress(0)
    status = st.empty()

    for i in range(5):
        status.text(f"Training iteration {i+1}/5...")
        progress.progress((i + 1) * 20)
        time.sleep(0.5)

        # Simulated metrics
        st.session_state.training_metrics.append({
            "iteration": len(st.session_state.training_metrics) + 1,
            "avg_outcome": 0.5 + 0.1 * (i + 1) + 0.05 * (i % 2),
            "buffer_size": (i + 1) * episodes,
            "policy_loss": 2.0 - 0.3 * i,
            "value_loss": 1.0 - 0.15 * i,
        })

    progress.empty()
    status.success("Training complete!")
    st.rerun()


def render_config_explorer():
    """Render the configuration explorer page."""
    st.header("Configuration Explorer")

    tab1, tab2, tab3 = st.tabs(["Graph Config", "MCTS Config", "Neural Config"])

    with tab1:
        st.subheader("Graph Configuration Presets")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Research Config**")
            config = create_research_config()
            st.json({
                "max_iterations": config.max_iterations,
                "consensus_threshold": config.confidence.consensus_threshold,
                "heuristic_weight": config.rollout_weights.heuristic_weight,
                "temperature": config.synthesis.temperature,
            })

        with col2:
            st.write("**Coding Config**")
            config = create_coding_config()
            st.json({
                "max_iterations": config.max_iterations,
                "consensus_threshold": config.confidence.consensus_threshold,
                "heuristic_weight": config.rollout_weights.heuristic_weight,
                "temperature": config.synthesis.temperature,
            })

        with col3:
            st.write("**Creative Config**")
            config = create_creative_config()
            st.json({
                "max_iterations": config.max_iterations,
                "consensus_threshold": config.confidence.consensus_threshold,
                "heuristic_weight": config.rollout_weights.heuristic_weight,
                "temperature": config.synthesis.temperature,
            })

    with tab2:
        st.subheader("MCTS Configuration Presets")

        for preset in ConfigPreset:
            with st.expander(f"{preset.value.title()} Preset"):
                config = create_preset_config(preset)
                st.json({
                    "num_iterations": config.num_iterations,
                    "exploration_weight": config.exploration_weight,
                    "progressive_widening_k": config.progressive_widening_k,
                    "progressive_widening_alpha": config.progressive_widening_alpha,
                    "max_rollout_depth": config.max_rollout_depth,
                    "early_termination_threshold": config.early_termination_threshold,
                })

    with tab3:
        st.subheader("Neural MCTS Configuration")

        if not NEURAL_AVAILABLE:
            st.warning("Neural MCTS not available")
        else:
            configs = {
                "Fast": get_fast_neural_config(),
                "Balanced": get_balanced_neural_config(),
                "Thorough": get_thorough_neural_config(),
                "AlphaZero": get_alphazero_config(),
            }

            for name, config in configs.items():
                with st.expander(f"{name} Preset"):
                    st.json({
                        "num_simulations": config.num_simulations,
                        "c_puct": config.c_puct,
                        "dirichlet_epsilon": config.dirichlet_epsilon,
                        "temperature_init": config.temperature_init,
                        "num_parallel_workers": config.num_parallel_workers,
                    })


def render_history():
    """Render query history page."""
    st.header("Query History")

    if not st.session_state.query_history:
        st.info("No queries yet. Try asking something!")
        return

    for i, item in enumerate(reversed(st.session_state.query_history)):
        with st.expander(f"{item['timestamp']} - Score: {item['consensus_score']:.2%}"):
            st.write(item["query"])

    if st.button("Clear History"):
        st.session_state.query_history = []
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    render_sidebar()

    # Main content
    st.title("LangGraph Multi-Agent MCTS")
    st.markdown("*Cognitive Architecture with Neural-Guided Tree Search*")

    # Navigation
    page = st.radio(
        "Navigate",
        ["Query", "Training", "Config Explorer", "History"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if page == "Query":
        render_main_query_interface()
    elif page == "Training":
        render_training_dashboard()
    elif page == "Config Explorer":
        render_config_explorer()
    elif page == "History":
        render_history()


if __name__ == "__main__":
    main()
