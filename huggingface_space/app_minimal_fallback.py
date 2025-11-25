#!/usr/bin/env python3
"""
Minimal Fallback App for Hugging Face Spaces
=============================================

This is a ZERO-DEPENDENCY fallback app that works with only core Python libraries.
Use this if the main app fails due to dependency conflicts.

This app demonstrates the framework concept without requiring:
- PyTorch
- Transformers
- PEFT
- sentence-transformers
- Any ML libraries

It provides a functional demo with mock agents and simulated routing.

VERSION: MINIMAL-FALLBACK-v1.0
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentResult:
    """Result from a single agent."""

    agent_name: str
    response: str
    confidence: float
    reasoning_steps: list
    execution_time_ms: float


@dataclass
class ControllerDecision:
    """Decision made by the meta-controller."""

    selected_agent: str
    confidence: float
    routing_probabilities: dict
    features_used: dict


class MinimalMetaController:
    """
    Minimal meta-controller using heuristic-based routing.
    No ML dependencies required.
    """

    def __init__(self, name: str = "HeuristicController"):
        self.name = name

    def extract_features(self, query: str) -> dict:
        """Extract simple heuristic features from query."""
        query_lower = query.lower()
        query_length = len(query)

        # Heuristic feature extraction
        has_multiple_questions = query.count("?") > 1
        has_comparison = any(word in query_lower for word in ["vs", "versus", "compare", "difference", "better"])
        has_optimization = any(
            word in query_lower for word in ["optimize", "best", "improve", "maximize", "minimize"]
        )
        has_technical = any(
            word in query_lower for word in ["algorithm", "code", "implement", "technical", "system"]
        )

        # Calculate confidence scores
        hrm_score = 0.5
        if has_multiple_questions:
            hrm_score += 0.3
        if has_technical:
            hrm_score += 0.1

        trm_score = 0.5
        if has_comparison:
            trm_score += 0.3
        if query_length > 100:
            trm_score += 0.1

        mcts_score = 0.5
        if has_optimization:
            mcts_score += 0.3
        if has_technical:
            mcts_score += 0.1

        # Normalize
        total = hrm_score + trm_score + mcts_score
        hrm_score /= total
        trm_score /= total
        mcts_score /= total

        return {
            "hrm_confidence": hrm_score,
            "trm_confidence": trm_score,
            "mcts_value": mcts_score,
            "query_length": query_length,
            "is_technical": has_technical,
            "has_optimization": has_optimization,
            "has_comparison": has_comparison,
        }

    def predict(self, query: str) -> tuple[str, float, dict]:
        """
        Predict which agent to route to.

        Returns:
            (selected_agent, confidence, probabilities)
        """
        features = self.extract_features(query)

        # Get scores
        probabilities = {
            "hrm": features["hrm_confidence"],
            "trm": features["trm_confidence"],
            "mcts": features["mcts_value"],
        }

        # Select agent with highest probability
        selected_agent = max(probabilities, key=probabilities.get)
        confidence = probabilities[selected_agent]

        return selected_agent, confidence, probabilities


class MinimalFramework:
    """Minimal multi-agent framework with zero ML dependencies."""

    def __init__(self):
        self.controller = MinimalMetaController()
        print(f"✅ Minimal Framework initialized (controller: {self.controller.name})")

    async def process_query(self, query: str) -> tuple[AgentResult, ControllerDecision]:
        """Process a query through the framework."""
        start_time = time.perf_counter()

        # Step 1: Get controller decision
        selected_agent, confidence, probabilities = self.controller.predict(query)
        features = self.controller.extract_features(query)

        # Step 2: Route to selected agent
        if selected_agent == "hrm":
            agent_result = await self._handle_hrm(query)
        elif selected_agent == "trm":
            agent_result = await self._handle_trm(query)
        else:
            agent_result = await self._handle_mcts(query)

        # Create controller decision
        controller_decision = ControllerDecision(
            selected_agent=selected_agent,
            confidence=confidence,
            routing_probabilities=probabilities,
            features_used=features,
        )

        total_time = (time.perf_counter() - start_time) * 1000
        agent_result.execution_time_ms = round(total_time, 2)

        return agent_result, controller_decision

    async def _handle_hrm(self, query: str) -> AgentResult:
        """Handle query with Hierarchical Reasoning Module."""
        await asyncio.sleep(0.1)

        steps = [
            "Decompose query into hierarchical subproblems",
            "Apply high-level reasoning (H-Module)",
            "Execute low-level refinement (L-Module)",
            "Synthesize hierarchical solution",
        ]

        response = f"[HRM Analysis] Breaking down the problem hierarchically:\n\n{query}\n\nThis approach decomposes complex problems into manageable hierarchical components."

        return AgentResult(
            agent_name="HRM (Hierarchical Reasoning)",
            response=response,
            confidence=0.85,
            reasoning_steps=steps,
            execution_time_ms=0.0,
        )

    async def _handle_trm(self, query: str) -> AgentResult:
        """Handle query with Tree Reasoning Module."""
        await asyncio.sleep(0.1)

        steps = [
            "Initialize solution state",
            "Recursive refinement iteration 1",
            "Recursive refinement iteration 2",
            "Convergence achieved - finalize",
        ]

        response = f"[TRM Analysis] Applying iterative refinement:\n\n{query}\n\nThis approach uses recursive refinement to progressively improve the solution."

        return AgentResult(
            agent_name="TRM (Iterative Refinement)",
            response=response,
            confidence=0.80,
            reasoning_steps=steps,
            execution_time_ms=0.0,
        )

    async def _handle_mcts(self, query: str) -> AgentResult:
        """Handle query with MCTS."""
        await asyncio.sleep(0.15)

        steps = [
            "Build search tree",
            "Selection: UCB1 exploration",
            "Expansion: Add promising nodes",
            "Simulation: Rollout evaluation",
            "Backpropagation: Update values",
        ]

        response = f"[MCTS Analysis] Strategic exploration via tree search:\n\n{query}\n\nThis approach uses Monte Carlo Tree Search to explore the solution space strategically."

        return AgentResult(
            agent_name="MCTS (Monte Carlo Tree Search)",
            response=response,
            confidence=0.88,
            reasoning_steps=steps,
            execution_time_ms=0.0,
        )


# Try to import Gradio, fall back to simple web interface
try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️ Gradio not available. Using text-only mode.")


def process_query_sync(query: str) -> tuple:
    """Synchronous wrapper for async processing."""
    if not query.strip():
        return ("Please enter a query.", {}, "", "", "", "")

    framework = MinimalFramework()
    agent_result, controller_decision = asyncio.run(framework.process_query(query))

    # Format outputs
    final_response = agent_result.response

    # Personality response (simple version)
    personality_response = f"""Hello! I've analyzed your query using our multi-agent framework.

{final_response}

This response was generated by the {agent_result.agent_name} with {agent_result.confidence:.1%} confidence.
"""

    # Controller decision visualization
    routing_viz = f"""### 🧠 Meta-Controller Decision

**Selected Agent:** `{controller_decision.selected_agent.upper()}`

**Confidence:** {controller_decision.confidence:.1%}

**Routing Probabilities:**
"""
    for agent, prob in controller_decision.routing_probabilities.items():
        bar = "█" * int(prob * 50)
        routing_viz += f"\n- **{agent.upper()}**: {prob:.1%} {bar}"

    # Agent details
    agent_details = {
        "agent": agent_result.agent_name,
        "confidence": f"{agent_result.confidence:.1%}",
        "reasoning_steps": agent_result.reasoning_steps,
        "execution_time_ms": agent_result.execution_time_ms,
    }

    # Features used
    features_viz = "### 📊 Features Used for Routing\n\n"
    for feature, value in controller_decision.features_used.items():
        if isinstance(value, float):
            features_viz += f"- **{feature}**: {value:.3f}\n"
        elif isinstance(value, bool):
            features_viz += f"- **{feature}**: {'Yes' if value else 'No'}\n"
        else:
            features_viz += f"- **{feature}**: {value}\n"

    # Metrics
    metrics = f"""**Execution Time:** {agent_result.execution_time_ms:.2f} ms
**Agent Confidence:** {agent_result.confidence:.1%}
**Controller:** Heuristic-based routing
"""

    return final_response, agent_details, routing_viz, features_viz, metrics, personality_response


# Example queries
EXAMPLE_QUERIES = [
    "What are the key factors to consider when choosing between microservices and monolithic architecture?",
    "How can we optimize a Python application that processes 10GB of log files daily?",
    "Compare the performance characteristics of B-trees vs LSM-trees for write-heavy workloads",
    "Design a distributed rate limiting system that handles 100k requests per second",
    "Explain the difference between supervised and unsupervised learning with examples",
]


def create_gradio_interface():
    """Create Gradio interface."""
    with gr.Blocks(
        title="LangGraph Multi-Agent MCTS - Minimal Fallback Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎯 LangGraph Multi-Agent MCTS Framework
            ## Minimal Fallback Demo (Zero ML Dependencies)

            ⚠️ **This is a minimal fallback version** that works without PyTorch, Transformers, or PEFT.
            It demonstrates the framework concept using heuristic-based routing.

            **Features:**
            - Heuristic-based meta-controller (no ML required)
            - Mock HRM, TRM, and MCTS agents
            - Fully functional demo with zero ML dependencies

            For the full version with trained neural meta-controllers, please resolve the dependency conflicts.

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Query", placeholder="Enter your question or reasoning task...", lines=4, max_lines=10
                )

                gr.Markdown("**Example Queries:**")
                example_dropdown = gr.Dropdown(choices=EXAMPLE_QUERIES, label="Select an example", interactive=True)

                def load_example(example):
                    return example

                example_dropdown.change(load_example, example_dropdown, query_input)

            with gr.Column(scale=1):
                gr.Markdown(
                    """
                **Controller Info:**
                - Type: Heuristic-based
                - No ML dependencies
                - Rule-based routing
                """
                )

        process_btn = gr.Button("🚀 Process Query", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Agent Response")
                final_response_output = gr.Textbox(label="Response", lines=6, interactive=False)

                gr.Markdown("### 🤝 Conversational Response")
                personality_output = gr.Textbox(label="Balanced Advisor Response", lines=8, interactive=False)

                gr.Markdown("### 📈 Performance Metrics")
                metrics_output = gr.Markdown()

            with gr.Column():
                routing_viz = gr.Markdown(label="Controller Decision")
                features_viz = gr.Markdown(label="Features")

        with gr.Accordion("🔍 Detailed Agent Information", open=False):
            agent_details_output = gr.JSON(label="Agent Execution Details")

        # Wire up the processing
        process_btn.click(
            fn=process_query_sync,
            inputs=[query_input],
            outputs=[
                final_response_output,
                agent_details_output,
                routing_viz,
                features_viz,
                metrics_output,
                personality_output,
            ],
        )

        gr.Markdown(
            """
            ---

            ### ℹ️ About This Minimal Demo

            This is a **fallback version** that works without any ML dependencies.
            It uses heuristic rules to route queries to agents.

            **Why use this version?**
            - Zero ML dependencies (no PyTorch, Transformers, PEFT)
            - Works on any system with Python 3.10+
            - Demonstrates the framework architecture
            - Useful for debugging dependency conflicts

            **To use the full version:**
            1. Resolve dependency conflicts (see verify_deployment.py)
            2. Use the main app.py with trained neural meta-controllers

            **Repository:** [GitHub - langgraph_multi_agent_mcts](https://github.com/ianshank/langgraph_multi_agent_mcts)

            ---
            *Minimal Fallback Version - No ML Dependencies Required*
            """
        )

    return demo


def main():
    """Main entry point."""
    print("=" * 80)
    print("LangGraph Multi-Agent MCTS Framework - Minimal Fallback Demo")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    if GRADIO_AVAILABLE:
        print("✅ Gradio available - launching web interface...")
        demo = create_gradio_interface()
        demo.launch(server_name="0.0.0.0", share=False, show_error=True)
    else:
        print("⚠️ Gradio not available - running in text mode")
        print()
        framework = MinimalFramework()

        # Interactive loop
        while True:
            print("\n" + "=" * 80)
            query = input("Enter your query (or 'quit' to exit): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            print()
            result, decision = asyncio.run(framework.process_query(query))

            print(f"Selected Agent: {decision.selected_agent.upper()}")
            print(f"Confidence: {decision.confidence:.1%}")
            print()
            print("Response:")
            print(result.response)
            print()
            print(f"Execution Time: {result.execution_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
