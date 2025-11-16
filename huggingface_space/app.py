"""
LangGraph Multi-Agent MCTS Framework - Hugging Face Spaces Demo

A proof-of-concept demonstration of multi-agent reasoning with Monte Carlo Tree Search.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Any

import gradio as gr
import numpy as np

# Demo-specific simplified implementations
from demo_src.mcts_demo import MCTSDemo
from demo_src.agents_demo import HRMAgent, TRMAgent
from demo_src.llm_mock import MockLLMClient, HuggingFaceClient


@dataclass
class AgentResult:
    """Result from a single agent."""
    agent_name: str
    response: str
    confidence: float
    reasoning_steps: list[str]
    execution_time_ms: float


@dataclass
class FrameworkResult:
    """Combined result from all agents."""
    query: str
    hrm_result: AgentResult | None
    trm_result: AgentResult | None
    mcts_result: dict | None
    consensus_score: float
    final_response: str
    total_time_ms: float
    metadata: dict


class MultiAgentFrameworkDemo:
    """Simplified multi-agent framework for Hugging Face Spaces demo."""

    def __init__(self, use_hf_inference: bool = False, hf_model: str = ""):
        """Initialize the demo framework.

        Args:
            use_hf_inference: Use Hugging Face Inference API instead of mock
            hf_model: Hugging Face model ID for inference
        """
        self.use_hf_inference = use_hf_inference
        self.hf_model = hf_model

        # Initialize components
        if use_hf_inference and hf_model:
            self.llm_client = HuggingFaceClient(model_id=hf_model)
        else:
            self.llm_client = MockLLMClient()

        self.hrm_agent = HRMAgent(self.llm_client)
        self.trm_agent = TRMAgent(self.llm_client)
        self.mcts = MCTSDemo()

    async def process_query(
        self,
        query: str,
        use_hrm: bool = True,
        use_trm: bool = True,
        use_mcts: bool = False,
        mcts_iterations: int = 25,
        exploration_weight: float = 1.414,
        seed: int | None = None
    ) -> FrameworkResult:
        """Process a query through the multi-agent framework.

        Args:
            query: The input query to process
            use_hrm: Enable Hierarchical Reasoning Module
            use_trm: Enable Tree Reasoning Module
            use_mcts: Enable Monte Carlo Tree Search
            mcts_iterations: Number of MCTS iterations
            exploration_weight: UCB1 exploration parameter
            seed: Random seed for reproducibility

        Returns:
            FrameworkResult with all agent outputs and consensus
        """
        start_time = time.perf_counter()

        hrm_result = None
        trm_result = None
        mcts_result = None

        # Run enabled agents
        tasks = []
        agent_names = []

        if use_hrm:
            tasks.append(self._run_hrm(query))
            agent_names.append("hrm")

        if use_trm:
            tasks.append(self._run_trm(query))
            agent_names.append("trm")

        if use_mcts:
            tasks.append(self._run_mcts(query, mcts_iterations, exploration_weight, seed))
            agent_names.append("mcts")

        # Execute agents concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(agent_names, results):
                if isinstance(result, Exception):
                    continue
                if name == "hrm":
                    hrm_result = result
                elif name == "trm":
                    trm_result = result
                elif name == "mcts":
                    mcts_result = result

        # Calculate consensus score
        consensus_score = self._calculate_consensus(hrm_result, trm_result, mcts_result)

        # Generate final synthesized response
        final_response = self._synthesize_response(
            query, hrm_result, trm_result, mcts_result, consensus_score
        )

        total_time = (time.perf_counter() - start_time) * 1000

        return FrameworkResult(
            query=query,
            hrm_result=hrm_result,
            trm_result=trm_result,
            mcts_result=mcts_result,
            consensus_score=consensus_score,
            final_response=final_response,
            total_time_ms=round(total_time, 2),
            metadata={
                "agents_used": agent_names,
                "mcts_config": {
                    "iterations": mcts_iterations,
                    "exploration_weight": exploration_weight,
                    "seed": seed
                } if use_mcts else None
            }
        )

    async def _run_hrm(self, query: str) -> AgentResult:
        """Run Hierarchical Reasoning Module."""
        start = time.perf_counter()
        result = await self.hrm_agent.process(query)
        elapsed = (time.perf_counter() - start) * 1000

        return AgentResult(
            agent_name="HRM (Hierarchical Reasoning)",
            response=result["response"],
            confidence=result["confidence"],
            reasoning_steps=result["steps"],
            execution_time_ms=round(elapsed, 2)
        )

    async def _run_trm(self, query: str) -> AgentResult:
        """Run Tree Reasoning Module."""
        start = time.perf_counter()
        result = await self.trm_agent.process(query)
        elapsed = (time.perf_counter() - start) * 1000

        return AgentResult(
            agent_name="TRM (Iterative Refinement)",
            response=result["response"],
            confidence=result["confidence"],
            reasoning_steps=result["steps"],
            execution_time_ms=round(elapsed, 2)
        )

    async def _run_mcts(
        self,
        query: str,
        iterations: int,
        exploration_weight: float,
        seed: int | None
    ) -> dict:
        """Run Monte Carlo Tree Search."""
        start = time.perf_counter()

        result = self.mcts.search(
            query=query,
            iterations=iterations,
            exploration_weight=exploration_weight,
            seed=seed
        )

        elapsed = (time.perf_counter() - start) * 1000
        result["execution_time_ms"] = round(elapsed, 2)

        return result

    def _calculate_consensus(
        self,
        hrm_result: AgentResult | None,
        trm_result: AgentResult | None,
        mcts_result: dict | None
    ) -> float:
        """Calculate agreement score between agents."""
        confidences = []

        if hrm_result:
            confidences.append(hrm_result.confidence)
        if trm_result:
            confidences.append(trm_result.confidence)
        if mcts_result:
            confidences.append(mcts_result.get("best_value", 0.5))

        if not confidences:
            return 0.0

        # Consensus is based on confidence alignment and average
        if len(confidences) == 1:
            return confidences[0]

        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)

        # Higher consensus when agents agree (low std) and are confident (high avg)
        agreement_factor = max(0, 1 - std_confidence * 2)
        consensus = avg_confidence * agreement_factor

        return round(min(1.0, consensus), 3)

    def _synthesize_response(
        self,
        query: str,
        hrm_result: AgentResult | None,
        trm_result: AgentResult | None,
        mcts_result: dict | None,
        consensus_score: float
    ) -> str:
        """Synthesize final response from all agent outputs."""
        parts = []

        if hrm_result and hrm_result.confidence > 0.5:
            parts.append(f"[HRM] {hrm_result.response}")

        if trm_result and trm_result.confidence > 0.5:
            parts.append(f"[TRM] {trm_result.response}")

        if mcts_result and mcts_result.get("best_value", 0) > 0.5:
            parts.append(f"[MCTS] Best path: {mcts_result.get('best_action', 'N/A')}")

        if not parts:
            return "Insufficient confidence from agents to provide a response."

        synthesis = " | ".join(parts)

        if consensus_score > 0.7:
            return f"HIGH CONSENSUS ({consensus_score:.1%}): {synthesis}"
        elif consensus_score > 0.4:
            return f"MODERATE CONSENSUS ({consensus_score:.1%}): {synthesis}"
        else:
            return f"LOW CONSENSUS ({consensus_score:.1%}): {synthesis}"


# Global framework instance
framework = None


def initialize_framework(use_hf: bool, model_id: str):
    """Initialize or reinitialize the framework."""
    global framework
    framework = MultiAgentFrameworkDemo(use_hf_inference=use_hf, hf_model=model_id)
    return "Framework initialized successfully!"


def process_query_sync(
    query: str,
    use_hrm: bool,
    use_trm: bool,
    use_mcts: bool,
    mcts_iterations: int,
    exploration_weight: float,
    seed: int
):
    """Synchronous wrapper for async processing."""
    global framework

    if framework is None:
        framework = MultiAgentFrameworkDemo()

    if not query.strip():
        return "Please enter a query.", {}, "", {}

    # Handle seed
    seed_value = seed if seed > 0 else None

    # Run async function
    result = asyncio.run(
        framework.process_query(
            query=query,
            use_hrm=use_hrm,
            use_trm=use_trm,
            use_mcts=use_mcts,
            mcts_iterations=int(mcts_iterations),
            exploration_weight=exploration_weight,
            seed=seed_value
        )
    )

    # Format outputs
    final_response = result.final_response

    # Agent details
    agent_details = {}
    if result.hrm_result:
        agent_details["HRM"] = {
            "response": result.hrm_result.response,
            "confidence": f"{result.hrm_result.confidence:.1%}",
            "reasoning_steps": result.hrm_result.reasoning_steps,
            "time_ms": result.hrm_result.execution_time_ms
        }

    if result.trm_result:
        agent_details["TRM"] = {
            "response": result.trm_result.response,
            "confidence": f"{result.trm_result.confidence:.1%}",
            "reasoning_steps": result.trm_result.reasoning_steps,
            "time_ms": result.trm_result.execution_time_ms
        }

    if result.mcts_result:
        agent_details["MCTS"] = result.mcts_result

    # Metrics
    metrics = f"""
**Consensus Score:** {result.consensus_score:.1%}
**Total Processing Time:** {result.total_time_ms:.2f} ms
**Agents Used:** {', '.join(result.metadata['agents_used'])}
"""

    # Full JSON result
    full_result = {
        "query": result.query,
        "final_response": result.final_response,
        "consensus_score": result.consensus_score,
        "total_time_ms": result.total_time_ms,
        "metadata": result.metadata,
        "agent_details": agent_details
    }

    return final_response, agent_details, metrics, full_result


def visualize_mcts_tree(mcts_result: dict) -> str:
    """Create ASCII visualization of MCTS tree."""
    if not mcts_result or "tree_visualization" not in mcts_result:
        return "No MCTS tree data available"

    return mcts_result["tree_visualization"]


# Example queries for demonstration
EXAMPLE_QUERIES = [
    "What are the key factors to consider when choosing between microservices and monolithic architecture?",
    "How can we optimize a Python application that processes 10GB of log files daily?",
    "What is the best approach to implement rate limiting in a distributed system?",
    "Should we use SQL or NoSQL database for a social media application with 1M users?",
    "How to design a fault-tolerant message queue system?",
]


# Gradio Interface
with gr.Blocks(
    title="LangGraph Multi-Agent MCTS Demo",
    theme=gr.themes.Soft(),
    css="""
    .agent-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin: 5px 0; }
    .consensus-high { color: #28a745; font-weight: bold; }
    .consensus-medium { color: #ffc107; font-weight: bold; }
    .consensus-low { color: #dc3545; font-weight: bold; }
    """
) as demo:
    gr.Markdown(
        """
        # LangGraph Multi-Agent MCTS Framework

        **Proof-of-Concept Demo** - Multi-agent reasoning with Monte Carlo Tree Search

        This demo showcases:
        - **HRM**: Hierarchical Reasoning Module - breaks down complex queries
        - **TRM**: Tree Reasoning Module - iterative refinement of responses
        - **MCTS**: Monte Carlo Tree Search - strategic exploration of solution space
        - **Consensus**: Agreement scoring between agents

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Query",
                placeholder="Enter your reasoning task or question...",
                lines=3,
                max_lines=10
            )

            gr.Markdown("**Example Queries:**")
            example_dropdown = gr.Dropdown(
                choices=EXAMPLE_QUERIES,
                label="Select an example",
                interactive=True
            )

            def load_example(example):
                return example

            example_dropdown.change(load_example, example_dropdown, query_input)

        with gr.Column(scale=1):
            gr.Markdown("**Agent Configuration**")
            use_hrm = gr.Checkbox(label="Enable HRM (Hierarchical)", value=True)
            use_trm = gr.Checkbox(label="Enable TRM (Iterative)", value=True)
            use_mcts = gr.Checkbox(label="Enable MCTS", value=False)

            gr.Markdown("**MCTS Parameters**")
            mcts_iterations = gr.Slider(
                minimum=10,
                maximum=100,
                value=25,
                step=5,
                label="Iterations",
                info="More iterations = better search, but slower"
            )
            exploration_weight = gr.Slider(
                minimum=0.1,
                maximum=3.0,
                value=1.414,
                step=0.1,
                label="Exploration Weight (C)",
                info="Higher = more exploration, Lower = more exploitation"
            )
            seed_input = gr.Number(
                label="Random Seed (0 for random)",
                value=0,
                precision=0
            )

    process_btn = gr.Button("Process Query", variant="primary", size="lg")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Final Response")
            final_response_output = gr.Textbox(
                label="Synthesized Response",
                lines=4,
                interactive=False
            )

            gr.Markdown("### Performance Metrics")
            metrics_output = gr.Markdown()

        with gr.Column():
            gr.Markdown("### Agent Details")
            agent_details_output = gr.JSON(label="Individual Agent Results")

    with gr.Accordion("Full JSON Result", open=False):
        full_result_output = gr.JSON(label="Complete Framework Output")

    # Wire up the processing
    process_btn.click(
        fn=process_query_sync,
        inputs=[
            query_input,
            use_hrm,
            use_trm,
            use_mcts,
            mcts_iterations,
            exploration_weight,
            seed_input
        ],
        outputs=[
            final_response_output,
            agent_details_output,
            metrics_output,
            full_result_output
        ]
    )

    gr.Markdown(
        """
        ---

        ### About This Demo

        This is a **proof-of-concept** demonstration of the LangGraph Multi-Agent MCTS Framework.

        **Features:**
        - Multi-agent orchestration with consensus scoring
        - Monte Carlo Tree Search for strategic reasoning
        - Configurable exploration vs exploitation trade-offs
        - Deterministic results with seeded randomness

        **Limitations (POC):**
        - Uses mock/simplified LLM responses (not production LLM)
        - Limited to demonstration scenarios
        - No persistent storage or RAG
        - Simplified MCTS implementation

        **Full Framework:** [GitHub Repository](https://github.com/ianshank/langgraph_multi_agent_mcts)

        ---
        *Built with LangGraph, Gradio, and Python*
        """
    )


if __name__ == "__main__":
    # Initialize with mock client for demo
    framework = MultiAgentFrameworkDemo(use_hf_inference=False)

    # Launch the demo
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
