"""
Neuro-Symbolic Reasoning Engine Demo.

Demonstrates the integration of:
1. Neural Planning (HRM with Tensor processing)
2. Symbolic Execution (LangGraph with MCTS)
3. Neuro-Symbolic Adapter (Bridge)
4. Training Workflow (PonderNet + Neural Planner)
5. Multiple Query Simulations with Varying Complexity
6. Comparative Analysis of Different Strategies

Run with:
    python examples/neuro_symbolic_demo.py --mode inference
    python examples/neuro_symbolic_demo.py --mode training
    python examples/neuro_symbolic_demo.py --mode full
    python examples/neuro_symbolic_demo.py --mode simulation
    python examples/neuro_symbolic_demo.py --mode benchmark
"""

import argparse
import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NeuroSymbolicDemo")


# ============================================================================
# Query Bank for Simulations
# ============================================================================

SIMULATION_QUERIES = {
    "simple": [
        ("What is the capital of France?", "Direct Answer", 1),
        ("Convert 100 Celsius to Fahrenheit", "Direct Answer", 1),
        ("What year did World War II end?", "Direct Answer", 1),
        ("Define machine learning in one sentence", "Direct Answer", 1),
        ("What is 2 + 2?", "Direct Answer", 1),
    ],
    "moderate": [
        ("Explain the difference between supervised and unsupervised learning", "Step by Step", 2),
        ("How does a transformer attention mechanism work?", "Step by Step", 3),
        ("Compare REST and GraphQL APIs", "Step by Step", 2),
        ("What are the pros and cons of microservices architecture?", "Deep Research", 3),
        ("How do I optimize a SQL query with multiple joins?", "Step by Step", 3),
    ],
    "complex": [
        ("Design a transition plan to Post-Quantum Cryptography", "Deep Research", 5),
        ("Build a distributed system for real-time fraud detection", "Deep Research", 6),
        ("Create a multi-agent AI system for autonomous code review", "Delegation", 5),
        ("Develop a strategy for migrating a monolith to microservices", "Deep Research", 5),
        ("Design a neural architecture search system for edge devices", "Deep Research", 6),
    ],
    "research": [
        ("Survey recent advances in large language model reasoning", "Deep Research", 7),
        ("Analyze the trade-offs between different consensus algorithms", "Deep Research", 6),
        ("Compare federated learning approaches for privacy-preserving ML", "Deep Research", 7),
        ("Review state-of-the-art methods in neural program synthesis", "Deep Research", 8),
        ("Evaluate different approaches to AI alignment and safety", "Deep Research", 8),
    ],
}

DOMAIN_CONTEXTS = {
    "cybersecurity": {
        "keywords": ["security", "cryptography", "attack", "threat", "vulnerability"],
        "boost_strategies": ["Deep Research", "Tool Use"],
    },
    "machine_learning": {
        "keywords": ["neural", "model", "training", "learning", "AI"],
        "boost_strategies": ["Step by Step", "Deep Research"],
    },
    "software_engineering": {
        "keywords": ["architecture", "design", "system", "code", "API"],
        "boost_strategies": ["Step by Step", "Delegation"],
    },
    "general": {
        "keywords": [],
        "boost_strategies": ["Direct Answer"],
    },
}


# ============================================================================
# Mock Components for Demo
# ============================================================================


class MockSubProblem:
    """Mock subproblem for demo without full model weights."""

    def __init__(self, level: int, description: str, confidence: float, complexity: float = 0.5):
        self.level = level
        self.description = description
        self.confidence = confidence
        self.complexity = complexity
        self.state = torch.randn(10)  # Mock tensor state


@dataclass
class MockPonderNetOutput:
    """Mock PonderNet output for demo."""

    halt_probs: list[torch.Tensor]
    step_outputs: list[torch.Tensor]
    halting_distribution: torch.Tensor
    expected_steps: float
    kl_divergence: torch.Tensor
    actual_steps: int = 0


@dataclass
class MockHRMOutput:
    """Mock HRM output for demo."""

    subproblems: list[MockSubProblem]
    total_ponder_cost: float = 0.5
    halt_step: int = 2
    convergence_path: list[float] = None
    final_state: Any = None
    ponder_output: MockPonderNetOutput = None
    processing_time_ms: float = 0.0


@dataclass
class SimulationResult:
    """Result from a single simulation run."""

    query: str
    expected_strategy: str
    predicted_strategy: str
    expected_steps: int
    actual_ponder_steps: float
    confidence: float
    processing_time_ms: float
    ponder_cost: float
    kl_divergence: float
    success: bool = True
    error_message: str = ""


@dataclass
class SimulationBatchResult:
    """Aggregated results from a batch of simulations."""

    category: str
    total_queries: int
    successful: int
    failed: int
    strategy_accuracy: float
    avg_ponder_steps: float
    avg_confidence: float
    avg_processing_time_ms: float
    avg_ponder_cost: float
    step_efficiency: float  # actual_steps / expected_steps ratio
    results: list[SimulationResult] = field(default_factory=list)


class MockNeuralHRM:
    """Simulates the PyTorch HRM Agent with PonderNet."""

    def __init__(self, use_ponder_net: bool = True, complexity_factor: float = 1.0):
        self.config = type(
            "Config",
            (),
            {
                "h_dim": 768,
                "max_ponder_steps": 16,
                "halt_threshold": 0.95,
            },
        )()
        self.use_ponder_net = use_ponder_net
        self.complexity_factor = complexity_factor
        self.ponder_net = type("PonderNet", (), {"lambda_p": 0.01})()
        self._call_count = 0

    def to(self, device):
        return self

    def train(self):
        pass

    def eval(self):
        pass

    def parameters(self):
        return iter([torch.randn(10, 10)])

    def get_parameter_count(self):
        return 1000000

    def _estimate_complexity(self, input_tensor: torch.Tensor, query: str = "") -> tuple[float, int]:
        """Estimate query complexity based on input tensor and text."""
        base_complexity = input_tensor.abs().mean().item()

        # Adjust based on query length and keywords
        if query:
            length_factor = min(len(query.split()) / 20, 1.5)
            keyword_boost = 0
            for domain, info in DOMAIN_CONTEXTS.items():
                if any(kw in query.lower() for kw in info["keywords"]):
                    keyword_boost = 0.3
                    break
            base_complexity = base_complexity * length_factor + keyword_boost

        # Map to steps with some randomness
        num_steps = min(int(base_complexity * 5 * self.complexity_factor) + 2, 8)
        num_steps = max(2, num_steps + random.randint(-1, 1))

        return base_complexity, num_steps

    def _generate_subproblems(self, query: str, num_steps: int) -> list[MockSubProblem]:
        """Generate contextual subproblems based on query."""
        templates = {
            "security": [
                "Analyze threat model and attack vectors",
                "Identify vulnerable components",
                "Research mitigation strategies",
                "Design security architecture",
                "Validate against known exploits",
                "Create implementation roadmap",
            ],
            "ml": [
                "Define problem formulation",
                "Analyze data requirements",
                "Select model architecture",
                "Design training pipeline",
                "Evaluate performance metrics",
                "Optimize for deployment",
            ],
            "software": [
                "Gather requirements",
                "Design system architecture",
                "Define component interfaces",
                "Plan implementation phases",
                "Create testing strategy",
                "Document deployment process",
            ],
            "general": [
                "Understand the question",
                "Research relevant information",
                "Analyze findings",
                "Synthesize response",
                "Validate accuracy",
            ],
        }

        # Select template based on query content
        template_key = "general"
        if any(kw in query.lower() for kw in ["security", "crypto", "attack", "threat"]):
            template_key = "security"
        elif any(kw in query.lower() for kw in ["model", "neural", "training", "learning"]):
            template_key = "ml"
        elif any(kw in query.lower() for kw in ["system", "architecture", "design", "api"]):
            template_key = "software"

        template = templates[template_key]
        subproblems = []
        for i in range(min(num_steps, len(template))):
            confidence = 0.9 - (i * 0.1) + random.uniform(-0.05, 0.05)
            complexity = (i + 1) / num_steps
            subproblems.append(
                MockSubProblem(
                    level=i,
                    description=template[i],
                    confidence=max(0.3, min(0.99, confidence)),
                    complexity=complexity,
                )
            )

        return subproblems

    def forward(
        self, input_tensor, return_decomposition=False, return_ponder_output=False, max_steps=None, query: str = ""
    ):
        """Forward pass with adaptive pondering."""
        start_time = time.time()
        self._call_count += 1

        # Estimate complexity and steps
        complexity, num_steps = self._estimate_complexity(input_tensor, query)
        if max_steps:
            num_steps = min(num_steps, max_steps)

        # Generate contextual subproblems
        subproblems = self._generate_subproblems(query, num_steps)

        # Create mock PonderNet output if requested
        ponder_output = None
        if return_ponder_output and self.use_ponder_net:
            halt_probs = [torch.sigmoid(torch.randn(1, 10) * (i + 1) / num_steps) for i in range(num_steps)]
            step_outputs = [torch.randn(1, 10, 768) for _ in range(num_steps)]
            halting_dist = torch.softmax(torch.randn(1, 10, num_steps), dim=-1)

            ponder_output = MockPonderNetOutput(
                halt_probs=halt_probs,
                step_outputs=step_outputs,
                halting_distribution=halting_dist,
                expected_steps=num_steps * 0.8 + random.uniform(-0.2, 0.2),
                kl_divergence=torch.tensor(0.05 * num_steps / 5),
                actual_steps=num_steps,
            )

        # Generate convergence path
        convergence_path = []
        for i in range(num_steps):
            progress = (i + 1) / num_steps
            noise = random.uniform(-0.05, 0.05)
            convergence_path.append(min(0.99, progress * 0.9 + noise))

        processing_time = (time.time() - start_time) * 1000

        return MockHRMOutput(
            subproblems=subproblems,
            convergence_path=convergence_path,
            halt_step=num_steps,
            total_ponder_cost=num_steps * 0.1 * complexity,
            ponder_output=ponder_output,
            processing_time_ms=processing_time + random.uniform(10, 50),
        )


class MockLLMAdapter:
    """Mock LLM adapter for demo with varied responses."""

    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms
        self._response_templates = {
            "analysis": "Analysis complete: {topic} shows {finding}.",
            "research": "Research findings: {topic} indicates {finding}.",
            "step": "Step completed: {action} with result {result}.",
            "synthesis": "Synthesized response: {summary}.",
        }

    async def generate(self, prompt: str, temperature: float = 0.0):
        await asyncio.sleep(self.latency_ms / 1000)

        # Generate contextual response
        text = "Generated response based on analysis."

        if "threat" in prompt.lower() or "attack" in prompt.lower():
            text = self._response_templates["analysis"].format(
                topic="security assessment", finding="multiple vectors requiring mitigation"
            )
        elif "research" in prompt.lower() or "survey" in prompt.lower():
            text = self._response_templates["research"].format(
                topic="literature review", finding="emerging consensus on best practices"
            )
        elif "design" in prompt.lower() or "architecture" in prompt.lower():
            text = self._response_templates["step"].format(
                action="architecture design", result="modular component structure"
            )
        elif "synthesize" in prompt.lower() or "final" in prompt.lower():
            text = self._response_templates["synthesis"].format(
                summary="comprehensive solution addressing all requirements"
            )

        logger.debug(f"[LLM] Generated: {text[:60]}...")
        return type("Response", (), {"text": text})()


class MockTRMAgent:
    """Mock TRM agent for demo with quality scoring."""

    def __init__(self, base_quality: float = 0.8):
        self.base_quality = base_quality

    async def process(self, query, rag_context=None):
        # Simulate quality based on query complexity
        complexity_penalty = min(len(query.split()) / 100, 0.2)
        quality = self.base_quality - complexity_penalty + random.uniform(-0.05, 0.1)
        quality = max(0.5, min(0.99, quality))

        logger.debug(f"[TRM] Processing with quality {quality:.2f}")
        return {"response": f"Refined response for: {query[:50]}...", "metadata": {"final_quality_score": quality}}


class MockNeuralPlanner:
    """Mock neural planner for strategy prediction."""

    def __init__(self, num_strategies: int = 5):
        self.strategies = ["Direct Answer", "Deep Research", "Step by Step", "Tool Use", "Delegation"]
        self.num_strategies = num_strategies

    def predict_strategy(self, query: str, complexity: float) -> tuple[str, float]:
        """Predict strategy based on query and complexity."""
        # Simple heuristic-based prediction
        if complexity < 0.3:
            strategy = "Direct Answer"
            confidence = 0.9
        elif complexity < 0.5:
            strategy = "Step by Step"
            confidence = 0.8
        elif complexity < 0.7:
            strategy = "Deep Research"
            confidence = 0.75
        else:
            strategy = random.choice(["Deep Research", "Delegation"])
            confidence = 0.7

        # Add some noise
        confidence += random.uniform(-0.1, 0.1)
        confidence = max(0.5, min(0.99, confidence))

        return strategy, confidence


# ============================================================================
# Inference Demo
# ============================================================================


async def run_inference_demo():
    """Demonstrate inference with Neuro-Symbolic adapter."""
    print("\n" + "=" * 60)
    print(" NEURO-SYMBOLIC REASONING ENGINE - INFERENCE DEMO")
    print("=" * 60 + "\n")

    # Import framework components
    from src.framework.adapters import NeuroSymbolicAdapter, create_neural_planner
    from src.framework.graph import IntegratedFramework
    from src.framework.mcts.config import ConfigPreset, create_preset_config

    # 1. Initialize Components
    print("1. Initializing Neural Agent (The Brain)...")
    neural_agent = MockNeuralHRM(use_ponder_net=True)
    trm_agent = MockTRMAgent()

    print("2. Initializing Neural Planner...")
    neural_planner = create_neural_planner(
        hidden_dim=512,
        num_strategies=5,
        max_steps=8,
        device="cpu",
    )
    print(f"   Planner parameters: {sum(p.numel() for p in neural_planner.parameters()):,}")

    print("3. Initializing LLM Adapter (The Hands)...")
    model_adapter = MockLLMAdapter()

    print("4. Building Integrated Framework...")
    framework = IntegratedFramework(
        model_adapter=model_adapter,
        logger=logger,
        mcts_config=create_preset_config(ConfigPreset.FAST),
        consensus_threshold=0.98,
        hrm_config={},
    )

    # Overwrite with our components
    framework.graph_builder.hrm_agent = neural_agent
    framework.graph_builder.trm_agent = trm_agent

    # Initialize adapter with both HRM and Planner
    framework.graph_builder.hrm_adapter = NeuroSymbolicAdapter(
        neural_agent=neural_agent,
        neural_planner=neural_planner,
        use_planner=False,  # Use HRM for this demo
    )

    # 5. Run Query
    query = "Design a transition plan to Post-Quantum Cryptography."
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    # Use the graph directly to see internal state changes
    app = framework.graph_builder.build_graph().compile()

    initial_state = {
        "query": query,
        "use_mcts": True,
        "use_rag": False,
        "iteration": 0,
        "max_iterations": 2,
        "agent_outputs": [],
    }

    print("\n--- Execution Flow ---\n")
    async for event in app.astream(initial_state):
        for node, state in event.items():
            print(f"Node Completed: {node}")

            if node == "hrm_agent":
                hrm_meta = state["hrm_results"]["metadata"]
                plan = hrm_meta.get("neural_plan", [])
                print(f"\n[Neuro-Symbolic Adapter] Neural Plan Detected ({len(plan)} steps):")
                for step in plan:
                    status_icon = "[R]" if step["strategy"] == "Deep Research" else "[D]"
                    print(f"  {status_icon} Level {step['level']}: {step['description']}")
                    print(f"     Strategy: {step['strategy']} (Confidence: {step['confidence']:.2f})")

                # Show PonderNet metrics if available
                if "expected_steps" in hrm_meta:
                    print(f"\n  [PonderNet] Expected steps: {hrm_meta['expected_steps']:.2f}")
                    print(f"  [PonderNet] KL divergence: {hrm_meta.get('kl_divergence', 0):.4f}")
                print("")

            elif node == "mcts_simulator":
                mcts_stats = state["mcts_stats"]
                best_action = state["mcts_best_action"]
                print(f"\n[MCTS Engine] Simulation Complete")
                print(f"  - Best Action: {best_action}")
                print(f"  - Simulations: {mcts_stats['iterations']}")
                print("")

            elif node == "synthesize":
                print(f"\n[Final Output]\n{state['final_response']}")

    print("\n" + "=" * 60)
    print(" INFERENCE DEMO COMPLETE")
    print("=" * 60)


# ============================================================================
# Training Demo
# ============================================================================


def run_training_demo():
    """Demonstrate training workflow for Neuro-Symbolic components."""
    print("\n" + "=" * 60)
    print(" NEURO-SYMBOLIC TRAINING DEMO")
    print("=" * 60 + "\n")

    from torch.utils.data import DataLoader

    from src.agents.hrm_agent import CurriculumPonderScheduler

    # Import training components
    from src.training.neuro_symbolic_trainer import (
        NeuroSymbolicConfig,
        NeuroSymbolicTrainer,
        collate_neuro_symbolic,
        create_synthetic_training_data,
    )
    from src.training.system_config import HRMConfig

    # 1. Configuration
    print("1. Setting up configuration...")
    config = NeuroSymbolicConfig(
        ponder_lambda=0.01,
        geometric_prior_p=0.5,
        max_ponder_steps=16,
        planner_hidden_dim=256,  # Smaller for demo
        planner_num_strategies=5,
        planner_max_steps=8,
        planner_num_layers=2,  # Smaller for demo
        learning_rate=1e-4,
        epochs=3,  # Short for demo
        batch_size=4,
        curriculum_warmup_epochs=1,
        curriculum_transition_epochs=2,
    )

    hrm_config = HRMConfig(
        h_dim=256,  # Smaller for demo
        l_dim=128,
        num_h_layers=1,
        num_l_layers=2,
        max_ponder_steps=config.max_ponder_steps,
    )

    print(f"   PonderNet lambda: {config.ponder_lambda}")
    print(f"   Geometric prior p: {config.geometric_prior_p}")
    print(f"   Max ponder steps: {config.max_ponder_steps}")
    print(f"   Planner strategies: {config.planner_num_strategies}")

    # 2. Create synthetic training data
    print("\n2. Creating synthetic training data...")
    dataset = create_synthetic_training_data(
        num_samples=100,
        embedding_dim=config.planner_hidden_dim,
        max_steps=config.planner_max_steps,
        num_strategies=config.planner_num_strategies,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_neuro_symbolic,
    )
    print(f"   Training samples: {len(dataset)}")
    print(f"   Batches per epoch: {len(train_dataloader)}")

    # 3. Initialize trainer
    print("\n3. Initializing Neuro-Symbolic Trainer...")
    trainer = NeuroSymbolicTrainer(
        config=config,
        hrm_config=hrm_config,
        device="cpu",  # Use CPU for demo
    )
    trainer.build_models()

    print(f"   HRM parameters: {trainer.hrm_agent.get_parameter_count():,}")
    print(f"   Planner parameters: {sum(p.numel() for p in trainer.neural_planner.parameters()):,}")

    # 4. Demonstrate curriculum scheduler
    print("\n4. Demonstrating Curriculum Learning...")
    curriculum = CurriculumPonderScheduler(
        warmup_epochs=config.curriculum_warmup_epochs,
        curriculum_epochs=config.curriculum_transition_epochs,
        initial_lambda_p=0.0,
        final_lambda_p=config.ponder_lambda,
        initial_max_steps=3,
        final_max_steps=config.max_ponder_steps,
    )

    print("   Curriculum progression:")
    for epoch in range(5):
        lambda_p = curriculum.get_lambda_p()
        max_steps = curriculum.get_max_steps()
        halt_thresh = curriculum.get_halt_threshold()
        print(f"   Epoch {epoch}: lambda_p={lambda_p:.4f}, max_steps={max_steps}, halt_threshold={halt_thresh:.2f}")
        curriculum.step()

    # 5. Run training
    print("\n5. Running Training Loop...")
    print("-" * 40)

    results = trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=config.epochs,
    )

    # 6. Display results
    print("\n6. Training Results:")
    print("-" * 40)
    for epoch, metrics in enumerate(results["train_history"]):
        print(f"   Epoch {epoch + 1}:")
        print(f"     Total Loss: {metrics['loss']:.4f}")
        print(f"     HRM Loss: {metrics['hrm_loss']:.4f}")
        print(f"     Planner Loss: {metrics['planner_loss']:.4f}")
        print(f"     Strategy Accuracy: {metrics['strategy_accuracy']:.2%}")
        print(f"     Ponder Cost: {metrics['ponder_cost']:.2f}")

    print("\n" + "=" * 60)
    print(" TRAINING DEMO COMPLETE")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")


# ============================================================================
# Simulation Demo - Multiple Query Scenarios
# ============================================================================


async def run_simulation_demo(num_queries_per_category: int = 5, verbose: bool = True):
    """Run simulations across multiple query categories."""
    print("\n" + "=" * 70)
    print(" NEURO-SYMBOLIC SIMULATION DEMO")
    print("=" * 70 + "\n")

    print("Running simulations across query complexity categories...")
    print(f"  - Queries per category: {num_queries_per_category}")
    print(f"  - Categories: {list(SIMULATION_QUERIES.keys())}")
    print("")

    # Initialize components
    neural_agent = MockNeuralHRM(use_ponder_net=True)
    neural_planner = MockNeuralPlanner(num_strategies=5)

    all_batch_results: list[SimulationBatchResult] = []

    for category, queries in SIMULATION_QUERIES.items():
        print(f"\n{'-' * 60}")
        print(f"  Category: {category.upper()}")
        print(f"{'-' * 60}")

        results: list[SimulationResult] = []
        selected_queries = queries[:num_queries_per_category]

        for i, (query, expected_strategy, expected_steps) in enumerate(selected_queries):
            # Create input tensor from query
            query_embedding = torch.randn(1, 10, 256)
            query_embedding = query_embedding * (len(query.split()) / 10)  # Scale by length

            # Get HRM output
            hrm_output = neural_agent.forward(
                query_embedding,
                return_ponder_output=True,
                query=query,
            )

            # Get strategy prediction
            complexity = hrm_output.total_ponder_cost / 0.8
            predicted_strategy, confidence = neural_planner.predict_strategy(query, complexity)

            # Calculate metrics
            actual_ponder = (
                hrm_output.ponder_output.expected_steps if hrm_output.ponder_output else hrm_output.halt_step
            )
            kl_div = hrm_output.ponder_output.kl_divergence.item() if hrm_output.ponder_output else 0.0

            result = SimulationResult(
                query=query,
                expected_strategy=expected_strategy,
                predicted_strategy=predicted_strategy,
                expected_steps=expected_steps,
                actual_ponder_steps=actual_ponder,
                confidence=confidence,
                processing_time_ms=hrm_output.processing_time_ms,
                ponder_cost=hrm_output.total_ponder_cost,
                kl_divergence=kl_div,
                success=True,
            )
            results.append(result)

            if verbose:
                match_icon = "[OK]" if predicted_strategy == expected_strategy else "[X]"
                print(f"\n  [{i + 1}] {query[:50]}...")
                print(f"      Strategy: {predicted_strategy} (expected: {expected_strategy}) {match_icon}")
                print(f"      Ponder: {actual_ponder:.1f} steps (expected: {expected_steps})")
                print(f"      Confidence: {confidence:.2%} | Cost: {hrm_output.total_ponder_cost:.3f}")

        # Aggregate batch results
        successful = sum(1 for r in results if r.predicted_strategy == r.expected_strategy)
        batch_result = SimulationBatchResult(
            category=category,
            total_queries=len(results),
            successful=successful,
            failed=len(results) - successful,
            strategy_accuracy=successful / len(results) if results else 0,
            avg_ponder_steps=sum(r.actual_ponder_steps for r in results) / len(results) if results else 0,
            avg_confidence=sum(r.confidence for r in results) / len(results) if results else 0,
            avg_processing_time_ms=sum(r.processing_time_ms for r in results) / len(results) if results else 0,
            avg_ponder_cost=sum(r.ponder_cost for r in results) / len(results) if results else 0,
            step_efficiency=sum(r.actual_ponder_steps / r.expected_steps for r in results) / len(results)
            if results
            else 0,
            results=results,
        )
        all_batch_results.append(batch_result)

    # Print summary
    print("\n" + "=" * 70)
    print(" SIMULATION SUMMARY")
    print("=" * 70)

    print("\n+" + "-" * 68 + "+")
    print(f"| {'Category':<12} | {'Accuracy':>10} | {'Avg Steps':>10} | {'Efficiency':>10} | {'Avg Cost':>10} |")
    print("+" + "-" * 68 + "+")

    for batch in all_batch_results:
        print(
            f"| {batch.category:<12} | {batch.strategy_accuracy:>9.1%} | {batch.avg_ponder_steps:>10.2f} | {batch.step_efficiency:>10.2f} | {batch.avg_ponder_cost:>10.3f} |"
        )

    print("+" + "-" * 68 + "+")

    # Overall stats
    total_queries = sum(b.total_queries for b in all_batch_results)
    total_successful = sum(b.successful for b in all_batch_results)
    overall_accuracy = total_successful / total_queries if total_queries else 0

    print(f"\n  Overall Strategy Accuracy: {overall_accuracy:.1%} ({total_successful}/{total_queries})")
    print(f"  Total Simulations: {total_queries}")

    return all_batch_results


# ============================================================================
# Benchmark Demo - Performance Comparison
# ============================================================================


async def run_benchmark_demo(iterations: int = 3):
    """Run benchmark comparing different configurations."""
    print("\n" + "=" * 70)
    print(" NEURO-SYMBOLIC BENCHMARK DEMO")
    print("=" * 70 + "\n")

    print("Benchmarking different PonderNet configurations...")

    configurations = [
        {"name": "Conservative", "complexity_factor": 0.5, "max_steps": 4},
        {"name": "Balanced", "complexity_factor": 1.0, "max_steps": 8},
        {"name": "Aggressive", "complexity_factor": 1.5, "max_steps": 12},
        {"name": "Adaptive", "complexity_factor": 1.2, "max_steps": 16},
    ]

    benchmark_results = []

    # Test queries of varying complexity
    test_queries = [
        ("Simple: What is 2+2?", 1),
        ("Moderate: Explain transformers", 3),
        ("Complex: Design a distributed system", 5),
        ("Research: Survey LLM reasoning methods", 7),
    ]

    for config in configurations:
        print(f"\n{'-' * 60}")
        print(f"  Configuration: {config['name']}")
        print(f"  Complexity Factor: {config['complexity_factor']}, Max Steps: {config['max_steps']}")
        print(f"{'-' * 60}")

        agent = MockNeuralHRM(use_ponder_net=True, complexity_factor=config["complexity_factor"])

        config_results = {
            "name": config["name"],
            "total_steps": 0,
            "total_cost": 0,
            "total_time_ms": 0,
            "query_results": [],
        }

        for iteration in range(iterations):
            for query, expected_steps in test_queries:
                query_embedding = torch.randn(1, 10, 256) * (expected_steps / 3)

                output = agent.forward(
                    query_embedding,
                    return_ponder_output=True,
                    max_steps=config["max_steps"],
                    query=query,
                )

                actual_steps = output.ponder_output.actual_steps if output.ponder_output else output.halt_step

                config_results["total_steps"] += actual_steps
                config_results["total_cost"] += output.total_ponder_cost
                config_results["total_time_ms"] += output.processing_time_ms

                if iteration == 0:  # Only print first iteration
                    efficiency = actual_steps / expected_steps
                    print(
                        f"  {query[:35]:<35} | Steps: {actual_steps:>2} (exp: {expected_steps}) | Eff: {efficiency:.2f}"
                    )

        num_total = len(test_queries) * iterations
        config_results["avg_steps"] = config_results["total_steps"] / num_total
        config_results["avg_cost"] = config_results["total_cost"] / num_total
        config_results["avg_time_ms"] = config_results["total_time_ms"] / num_total

        benchmark_results.append(config_results)

    # Print benchmark summary
    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS")
    print("=" * 70)

    print("\n+" + "-" * 58 + "+")
    print(f"| {'Configuration':<14} | {'Avg Steps':>10} | {'Avg Cost':>10} | {'Avg Time':>12} |")
    print("+" + "-" * 58 + "+")

    for result in benchmark_results:
        print(
            f"| {result['name']:<14} | {result['avg_steps']:>10.2f} | {result['avg_cost']:>10.3f} | {result['avg_time_ms']:>10.2f}ms |"
        )

    print("+" + "-" * 58 + "+")

    # Find best configuration
    best_efficiency = min(benchmark_results, key=lambda x: x["avg_cost"] / max(x["avg_steps"], 1))
    print(f"\n  Most Efficient: {best_efficiency['name']} (cost/step ratio)")

    return benchmark_results


# ============================================================================
# Curriculum Learning Visualization Demo
# ============================================================================


def run_curriculum_visualization_demo():
    """Visualize curriculum learning progression."""
    print("\n" + "=" * 70)
    print(" CURRICULUM LEARNING VISUALIZATION")
    print("=" * 70 + "\n")

    from src.agents.hrm_agent import CurriculumPonderScheduler

    # Create curriculum scheduler
    curriculum = CurriculumPonderScheduler(
        warmup_epochs=5,
        curriculum_epochs=10,
        initial_lambda_p=0.0,
        final_lambda_p=0.05,
        initial_max_steps=3,
        final_max_steps=16,
    )

    print("  Curriculum Progression Over 20 Epochs:")
    print("  " + "-" * 56)

    # Header
    print(f"  {'Epoch':>6} | {'lambda_p':>8} | {'Max Steps':>10} | {'Halt Thresh':>12} | {'Phase':>10}")
    print("  " + "-" * 56)

    for epoch in range(20):
        lambda_p = curriculum.get_lambda_p()
        max_steps = curriculum.get_max_steps()
        halt_thresh = curriculum.get_halt_threshold()

        # Determine phase
        if epoch < 5:
            phase = "Warmup"
        elif epoch < 15:
            phase = "Curriculum"
        else:
            phase = "Final"

        # Visual bar for lambda_p
        bar_len = int(lambda_p / 0.05 * 10)
        bar = "#" * bar_len + "." * (10 - bar_len)

        print(f"  {epoch:>6} | {lambda_p:>8.4f} | {max_steps:>10} | {halt_thresh:>12.2f} | {phase:>10} | {bar}")

        curriculum.step()

    print("  " + "-" * 56)
    print("\n  Legend: lambda_p = ponder cost weight, higher = more regularization")


# ============================================================================
# Multi-Agent Coordination Demo
# ============================================================================


async def run_multi_agent_demo():
    """Demonstrate multi-agent coordination with neural planning."""
    print("\n" + "=" * 70)
    print(" MULTI-AGENT COORDINATION DEMO")
    print("=" * 70 + "\n")

    print("Simulating coordinated agent execution with neural planning...\n")

    # Initialize agents
    hrm_agent = MockNeuralHRM(use_ponder_net=True)
    trm_agent = MockTRMAgent(base_quality=0.85)
    llm_adapter = MockLLMAdapter(latency_ms=30)
    planner = MockNeuralPlanner()

    # Complex multi-step query
    query = "Design and implement a secure, scalable microservices architecture for real-time data processing"

    print(f"  Query: {query}\n")
    print("  " + "-" * 60)

    # Step 1: Neural Planning
    print("\n  [Phase 1: Neural Planning]")
    query_embedding = torch.randn(1, 10, 256) * 1.5
    hrm_output = hrm_agent.forward(query_embedding, return_ponder_output=True, query=query)

    strategy, confidence = planner.predict_strategy(query, hrm_output.total_ponder_cost)
    print(f"    Strategy: {strategy} (confidence: {confidence:.2%})")
    print(f"    Ponder Steps: {hrm_output.halt_step}")
    print(f"    Subproblems: {len(hrm_output.subproblems)}")

    # Step 2: Hierarchical Decomposition
    print("\n  [Phase 2: Hierarchical Decomposition]")
    for i, subproblem in enumerate(hrm_output.subproblems):
        print(f"    {i + 1}. {subproblem.description}")
        print(
            f"       Level: {subproblem.level} | Confidence: {subproblem.confidence:.2%} | Complexity: {subproblem.complexity:.2f}"
        )

    # Step 3: Parallel Agent Execution
    print("\n  [Phase 3: Parallel Agent Execution]")

    async def execute_subproblem(idx: int, subproblem: MockSubProblem):
        start = time.time()

        # LLM generates response
        response = await llm_adapter.generate(f"Execute: {subproblem.description}")

        # TRM refines
        refined = await trm_agent.process(subproblem.description)

        elapsed = (time.time() - start) * 1000
        return {
            "idx": idx,
            "description": subproblem.description,
            "response": response.text,
            "quality": refined["metadata"]["final_quality_score"],
            "time_ms": elapsed,
        }

    # Execute in parallel
    tasks = [execute_subproblem(i, sp) for i, sp in enumerate(hrm_output.subproblems)]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"    Subproblem {result['idx'] + 1}: Quality={result['quality']:.2%}, Time={result['time_ms']:.1f}ms")

    # Step 4: Synthesis
    print("\n  [Phase 4: Synthesis]")
    total_quality = sum(r["quality"] for r in results) / len(results)
    total_time = sum(r["time_ms"] for r in results)

    print(f"    Average Quality: {total_quality:.2%}")
    print(f"    Total Execution Time: {total_time:.1f}ms")
    print(f"    Parallelization Speedup: {len(results)}x theoretical")

    # Convergence visualization
    print("\n  [Convergence Path]")
    path = hrm_output.convergence_path
    for i, conv in enumerate(path):
        bar_len = int(conv * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"    Step {i + 1}: [{bar}] {conv:.1%}")

    print("\n  " + "-" * 60)
    print("  Multi-agent coordination complete!")

    return {
        "strategy": strategy,
        "confidence": confidence,
        "subproblems": len(hrm_output.subproblems),
        "total_quality": total_quality,
        "total_time_ms": total_time,
    }


# ============================================================================
# Full Demo (Inference + Training)
# ============================================================================


async def run_full_demo():
    """Run both inference and training demos."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " NEURO-SYMBOLIC REASONING ENGINE - FULL DEMO".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    print("This demo showcases the complete Neuro-Symbolic training pipeline:")
    print("  - PonderNet: Learns WHEN to stop thinking (adaptive computation)")
    print("  - Neural Planner: Learns WHAT strategy to use (decomposition)")
    print("  - Curriculum Learning: Gradual transition from fixed to adaptive")
    print("  - Integration with HRM and LangGraph\n")

    # Part 1: Training
    print("\n" + "=" * 70)
    print(" PART 1: TRAINING")
    print("=" * 70)
    run_training_demo()

    # Part 2: Inference
    print("\n" + "=" * 70)
    print(" PART 2: INFERENCE")
    print("=" * 70)
    await run_inference_demo()

    print("\n" + "#" * 70)
    print("#" + " FULL DEMO COMPLETE".center(68) + "#")
    print("#" * 70)


# ============================================================================
# Extended Full Demo (All Components)
# ============================================================================


async def run_extended_demo():
    """Run all demo components including simulations and benchmarks."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " NEURO-SYMBOLIC EXTENDED DEMO".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    print("This extended demo includes:")
    print("  1. Training Demo - PonderNet & Neural Planner training")
    print("  2. Inference Demo - LangGraph integration")
    print("  3. Simulation Demo - Multiple query scenarios")
    print("  4. Benchmark Demo - Configuration comparison")
    print("  5. Curriculum Demo - Learning progression")
    print("  6. Multi-Agent Demo - Coordinated execution")
    print("")

    # Part 1: Training
    print("\n" + "=" * 70)
    print(" PART 1: TRAINING")
    print("=" * 70)
    run_training_demo()

    # Part 2: Inference
    print("\n" + "=" * 70)
    print(" PART 2: INFERENCE")
    print("=" * 70)
    await run_inference_demo()

    # Part 3: Simulations
    print("\n" + "=" * 70)
    print(" PART 3: SIMULATIONS")
    print("=" * 70)
    await run_simulation_demo(num_queries_per_category=3, verbose=True)

    # Part 4: Benchmarks
    print("\n" + "=" * 70)
    print(" PART 4: BENCHMARKS")
    print("=" * 70)
    await run_benchmark_demo(iterations=2)

    # Part 5: Curriculum Visualization
    print("\n" + "=" * 70)
    print(" PART 5: CURRICULUM LEARNING")
    print("=" * 70)
    run_curriculum_visualization_demo()

    # Part 6: Multi-Agent Coordination
    print("\n" + "=" * 70)
    print(" PART 6: MULTI-AGENT COORDINATION")
    print("=" * 70)
    await run_multi_agent_demo()

    print("\n" + "#" * 70)
    print("#" + " EXTENDED DEMO COMPLETE".center(68) + "#")
    print("#" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Reasoning Engine Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/neuro_symbolic_demo.py --mode inference
  python examples/neuro_symbolic_demo.py --mode training
  python examples/neuro_symbolic_demo.py --mode full
  python examples/neuro_symbolic_demo.py --mode simulation
  python examples/neuro_symbolic_demo.py --mode benchmark
  python examples/neuro_symbolic_demo.py --mode curriculum
  python examples/neuro_symbolic_demo.py --mode multiagent
  python examples/neuro_symbolic_demo.py --mode extended
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "training", "full", "simulation", "benchmark", "curriculum", "multiagent", "extended"],
        default="full",
        help="Demo mode (default: full)",
    )
    parser.add_argument(
        "--queries", type=int, default=5, help="Number of queries per category for simulation mode (default: 5)"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations for benchmark mode (default: 3)"
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output (default: True)")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    args = parser.parse_args()
    verbose = not args.quiet and args.verbose

    if args.mode == "inference":
        asyncio.run(run_inference_demo())
    elif args.mode == "training":
        run_training_demo()
    elif args.mode == "simulation":
        asyncio.run(run_simulation_demo(num_queries_per_category=args.queries, verbose=verbose))
    elif args.mode == "benchmark":
        asyncio.run(run_benchmark_demo(iterations=args.iterations))
    elif args.mode == "curriculum":
        run_curriculum_visualization_demo()
    elif args.mode == "multiagent":
        asyncio.run(run_multi_agent_demo())
    elif args.mode == "extended":
        asyncio.run(run_extended_demo())
    else:
        asyncio.run(run_full_demo())


if __name__ == "__main__":
    main()
