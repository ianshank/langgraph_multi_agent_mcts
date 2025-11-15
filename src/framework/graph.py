"""
LangGraph Integration Module - Extract graph building with new MCTS core integration.

Provides:
- Graph building extracted from LangGraphMultiAgentFramework
- Integration with new deterministic MCTS core
- Backward compatibility with original process() signature
- Support for parallel HRM/TRM execution
"""

from __future__ import annotations
import asyncio
import time
from typing import TypedDict, Annotated, List, Dict, Optional, Any
from typing_extensions import NotRequired
import operator

# LangGraph imports (these would be installed dependencies)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Stubs for development without LangGraph installed
    StateGraph = None
    END = "END"
    MemorySaver = None

# Import new MCTS modules
from .mcts.core import MCTSEngine, MCTSNode, MCTSState
from .mcts.config import MCTSConfig, create_preset_config, ConfigPreset
from .mcts.policies import (
    SelectionPolicy,
    HybridRolloutPolicy,
    RandomRolloutPolicy,
    GreedyRolloutPolicy,
)
from .mcts.experiments import ExperimentTracker


class AgentState(TypedDict):
    """Shared state for LangGraph agent framework."""

    # Input
    query: str
    use_mcts: bool
    use_rag: bool

    # RAG context
    rag_context: NotRequired[str]
    retrieved_docs: NotRequired[List[Dict]]

    # Agent results
    hrm_results: NotRequired[Dict]
    trm_results: NotRequired[Dict]
    agent_outputs: Annotated[List[Dict], operator.add]

    # MCTS simulation (updated for new core)
    mcts_root: NotRequired[Any]  # MCTSNode
    mcts_iterations: NotRequired[int]
    mcts_best_action: NotRequired[str]
    mcts_stats: NotRequired[Dict]
    mcts_config: NotRequired[Dict]

    # Evaluation
    confidence_scores: NotRequired[Dict[str, float]]
    consensus_reached: NotRequired[bool]
    consensus_score: NotRequired[float]

    # Control flow
    iteration: int
    max_iterations: int

    # Output
    final_response: NotRequired[str]
    metadata: NotRequired[Dict]


class GraphBuilder:
    """
    Builds and configures the LangGraph state machine for multi-agent orchestration.

    Extracts graph building logic from LangGraphMultiAgentFramework for modularity.
    """

    def __init__(
        self,
        hrm_agent,
        trm_agent,
        model_adapter,
        logger,
        vector_store=None,
        mcts_config: Optional[MCTSConfig] = None,
        top_k_retrieval: int = 5,
        max_iterations: int = 3,
        consensus_threshold: float = 0.75,
        enable_parallel_agents: bool = True,
    ):
        """
        Initialize graph builder.

        Args:
            hrm_agent: HRM agent instance
            trm_agent: TRM agent instance
            model_adapter: Model adapter for LLM calls
            logger: Logger instance
            vector_store: Optional vector store for RAG
            mcts_config: MCTS configuration (uses balanced preset if None)
            top_k_retrieval: Number of documents for RAG
            max_iterations: Maximum agent iterations
            consensus_threshold: Threshold for consensus
            enable_parallel_agents: Run HRM/TRM in parallel
        """
        self.hrm_agent = hrm_agent
        self.trm_agent = trm_agent
        self.model_adapter = model_adapter
        self.logger = logger
        self.vector_store = vector_store
        self.top_k_retrieval = top_k_retrieval
        self.max_iterations = max_iterations
        self.consensus_threshold = consensus_threshold
        self.enable_parallel_agents = enable_parallel_agents

        # MCTS configuration
        self.mcts_config = mcts_config or create_preset_config(ConfigPreset.BALANCED)

        # MCTS engine with deterministic behavior
        self.mcts_engine = MCTSEngine(
            seed=self.mcts_config.seed,
            exploration_weight=self.mcts_config.exploration_weight,
            progressive_widening_k=self.mcts_config.progressive_widening_k,
            progressive_widening_alpha=self.mcts_config.progressive_widening_alpha,
            max_parallel_rollouts=self.mcts_config.max_parallel_rollouts,
            cache_size_limit=self.mcts_config.cache_size_limit,
        )

        # Experiment tracking
        self.experiment_tracker = ExperimentTracker(name="langgraph_mcts")

    def build_graph(self) -> StateGraph:
        """
        Build LangGraph state machine.

        Returns:
            Configured StateGraph
        """
        if StateGraph is None:
            raise ImportError("LangGraph not installed. Install with: pip install langgraph")

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("entry", self._entry_node)
        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("route_decision", self._route_decision_node)
        workflow.add_node("parallel_agents", self._parallel_agents_node)
        workflow.add_node("hrm_agent", self._hrm_agent_node)
        workflow.add_node("trm_agent", self._trm_agent_node)
        workflow.add_node("mcts_simulator", self._mcts_simulator_node)
        workflow.add_node("aggregate_results", self._aggregate_results_node)
        workflow.add_node("evaluate_consensus", self._evaluate_consensus_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("entry")
        workflow.add_edge("entry", "retrieve_context")
        workflow.add_edge("retrieve_context", "route_decision")

        # Conditional routing
        workflow.add_conditional_edges(
            "route_decision",
            self._route_to_agents,
            {
                "parallel": "parallel_agents",
                "hrm": "hrm_agent",
                "trm": "trm_agent",
                "mcts": "mcts_simulator",
                "aggregate": "aggregate_results",
            },
        )

        # Parallel agents to aggregation
        workflow.add_edge("parallel_agents", "aggregate_results")

        # Sequential agent nodes
        workflow.add_edge("hrm_agent", "aggregate_results")
        workflow.add_edge("trm_agent", "aggregate_results")
        workflow.add_edge("mcts_simulator", "aggregate_results")

        # Aggregation to evaluation
        workflow.add_edge("aggregate_results", "evaluate_consensus")

        # Conditional consensus check
        workflow.add_conditional_edges(
            "evaluate_consensus",
            self._check_consensus,
            {
                "synthesize": "synthesize",
                "iterate": "route_decision",
            },
        )

        # Synthesis to end
        workflow.add_edge("synthesize", END)

        return workflow

    def _entry_node(self, state: AgentState) -> Dict:
        """Initialize state and parse query."""
        self.logger.info(f"Entry node: {state['query'][:100]}")
        return {
            "iteration": 0,
            "agent_outputs": [],
            "mcts_config": self.mcts_config.to_dict(),
        }

    def _retrieve_context_node(self, state: AgentState) -> Dict:
        """Retrieve context from vector store using RAG."""
        if not state.get("use_rag", True) or not self.vector_store:
            return {"rag_context": ""}

        query = state["query"]

        # Retrieve documents
        docs = self.vector_store.similarity_search(query, k=self.top_k_retrieval)

        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])

        self.logger.info(f"Retrieved {len(docs)} documents")

        return {
            "rag_context": context,
            "retrieved_docs": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ],
        }

    def _route_decision_node(self, state: AgentState) -> Dict:
        """Prepare routing decision."""
        return {}

    def _route_to_agents(self, state: AgentState) -> str:
        """Route to appropriate agent based on state."""
        iteration = state.get("iteration", 0)

        # First iteration: run HRM and TRM
        if iteration == 0:
            if self.enable_parallel_agents:
                if "hrm_results" not in state and "trm_results" not in state:
                    return "parallel"
            else:
                if "hrm_results" not in state:
                    return "hrm"
                elif "trm_results" not in state:
                    return "trm"

        # Run MCTS if enabled and not yet done
        if state.get("use_mcts", False) and "mcts_stats" not in state:
            return "mcts"

        return "aggregate"

    async def _parallel_agents_node(self, state: AgentState) -> Dict:
        """Execute HRM and TRM agents in parallel."""
        self.logger.info("Executing HRM and TRM agents in parallel")

        # Run both agents concurrently
        hrm_task = asyncio.create_task(
            self.hrm_agent.process(
                query=state["query"],
                rag_context=state.get("rag_context"),
            )
        )

        trm_task = asyncio.create_task(
            self.trm_agent.process(
                query=state["query"],
                rag_context=state.get("rag_context"),
            )
        )

        # Await both results
        hrm_result, trm_result = await asyncio.gather(hrm_task, trm_task)

        # Combine outputs
        return {
            "hrm_results": {
                "response": hrm_result["response"],
                "metadata": hrm_result["metadata"],
            },
            "trm_results": {
                "response": trm_result["response"],
                "metadata": trm_result["metadata"],
            },
            "agent_outputs": [
                {
                    "agent": "hrm",
                    "response": hrm_result["response"],
                    "confidence": hrm_result["metadata"].get(
                        "decomposition_quality_score", 0.7
                    ),
                },
                {
                    "agent": "trm",
                    "response": trm_result["response"],
                    "confidence": trm_result["metadata"].get(
                        "final_quality_score", 0.7
                    ),
                },
            ],
        }

    async def _hrm_agent_node(self, state: AgentState) -> Dict:
        """Execute HRM agent."""
        self.logger.info("Executing HRM agent")

        result = await self.hrm_agent.process(
            query=state["query"],
            rag_context=state.get("rag_context"),
        )

        return {
            "hrm_results": {
                "response": result["response"],
                "metadata": result["metadata"],
            },
            "agent_outputs": [
                {
                    "agent": "hrm",
                    "response": result["response"],
                    "confidence": result["metadata"].get(
                        "decomposition_quality_score", 0.7
                    ),
                }
            ],
        }

    async def _trm_agent_node(self, state: AgentState) -> Dict:
        """Execute TRM agent."""
        self.logger.info("Executing TRM agent")

        result = await self.trm_agent.process(
            query=state["query"],
            rag_context=state.get("rag_context"),
        )

        return {
            "trm_results": {
                "response": result["response"],
                "metadata": result["metadata"],
            },
            "agent_outputs": [
                {
                    "agent": "trm",
                    "response": result["response"],
                    "confidence": result["metadata"].get("final_quality_score", 0.7),
                }
            ],
        }

    async def _mcts_simulator_node(self, state: AgentState) -> Dict:
        """Execute MCTS simulation using new deterministic engine."""
        self.logger.info("Executing MCTS simulation with deterministic engine")

        start_time = time.perf_counter()

        # Reset engine for this simulation
        self.mcts_engine.clear_cache()

        # Create root state
        root_state = MCTSState(
            state_id="root",
            features={
                "query": state["query"][:100],  # Truncate for hashing
                "has_hrm": "hrm_results" in state,
                "has_trm": "trm_results" in state,
            },
        )

        root = MCTSNode(
            state=root_state,
            rng=self.mcts_engine.rng,
        )

        # Define action generator based on domain
        def action_generator(mcts_state: MCTSState) -> List[str]:
            """Generate available actions for state."""
            depth = len(mcts_state.state_id.split("_")) - 1

            if depth == 0:
                # Root level actions
                return ["action_A", "action_B", "action_C", "action_D"]
            elif depth < self.mcts_config.max_tree_depth:
                # Subsequent actions
                return ["continue", "refine", "fallback", "escalate"]
            else:
                return []  # Terminal

        # Define state transition
        def state_transition(mcts_state: MCTSState, action: str) -> MCTSState:
            """Compute next state from action."""
            new_id = f"{mcts_state.state_id}_{action}"
            new_features = mcts_state.features.copy()
            new_features["last_action"] = action
            new_features["depth"] = len(new_id.split("_")) - 1
            return MCTSState(state_id=new_id, features=new_features)

        # Create rollout policy using agent results
        def heuristic_fn(mcts_state: MCTSState) -> float:
            """Evaluate state using agent confidence."""
            base = 0.5

            # Bias based on agent confidence
            if state.get("hrm_results"):
                hrm_conf = state["hrm_results"]["metadata"].get(
                    "decomposition_quality_score", 0.5
                )
                base += hrm_conf * 0.2

            if state.get("trm_results"):
                trm_conf = state["trm_results"]["metadata"].get(
                    "final_quality_score", 0.5
                )
                base += trm_conf * 0.2

            return min(base, 1.0)

        rollout_policy = HybridRolloutPolicy(
            heuristic_fn=heuristic_fn,
            heuristic_weight=0.7,
            random_weight=0.3,
        )

        # Run MCTS search
        best_action, stats = await self.mcts_engine.search(
            root=root,
            num_iterations=self.mcts_config.num_iterations,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
            max_rollout_depth=self.mcts_config.max_rollout_depth,
            selection_policy=self.mcts_config.selection_policy,
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Compute tree statistics
        tree_depth = self.mcts_engine.get_tree_depth(root)
        tree_node_count = self.mcts_engine.count_nodes(root)

        # Track experiment
        self.experiment_tracker.create_result(
            experiment_id=f"mcts_{int(time.time())}",
            config=self.mcts_config,
            mcts_stats=stats,
            execution_time_ms=execution_time_ms,
            tree_depth=tree_depth,
            tree_node_count=tree_node_count,
            metadata={
                "query": state["query"][:100],
                "has_rag": state.get("use_rag", False),
            },
        )

        self.logger.info(
            f"MCTS complete: best_action={best_action}, "
            f"iterations={stats['iterations']}, "
            f"cache_hit_rate={stats['cache_hit_rate']:.2%}"
        )

        return {
            "mcts_root": root,
            "mcts_best_action": best_action,
            "mcts_stats": stats,
            "agent_outputs": [
                {
                    "agent": "mcts",
                    "response": (
                        f"Simulated {stats['iterations']} scenarios with "
                        f"seed {self.mcts_config.seed}. "
                        f"Recommended action: {best_action} "
                        f"(visits={stats['best_action_visits']}, "
                        f"value={stats['best_action_value']:.3f})"
                    ),
                    "confidence": min(
                        stats["best_action_visits"] / stats["iterations"]
                        if stats["iterations"] > 0
                        else 0.5,
                        1.0,
                    ),
                }
            ],
        }

    def _aggregate_results_node(self, state: AgentState) -> Dict:
        """Aggregate results from all agents."""
        self.logger.info("Aggregating agent results")

        agent_outputs = state.get("agent_outputs", [])

        confidence_scores = {
            output["agent"]: output["confidence"] for output in agent_outputs
        }

        return {"confidence_scores": confidence_scores}

    def _evaluate_consensus_node(self, state: AgentState) -> Dict:
        """Evaluate consensus among agents."""
        agent_outputs = state.get("agent_outputs", [])

        if len(agent_outputs) < 2:
            return {
                "consensus_reached": True,
                "consensus_score": 1.0,
            }

        avg_confidence = sum(o["confidence"] for o in agent_outputs) / len(
            agent_outputs
        )

        consensus_reached = avg_confidence >= self.consensus_threshold

        self.logger.info(
            f"Consensus: {consensus_reached} (score={avg_confidence:.2f})"
        )

        return {
            "consensus_reached": consensus_reached,
            "consensus_score": avg_confidence,
        }

    def _check_consensus(self, state: AgentState) -> str:
        """Check if consensus reached or need more iterations."""
        if state.get("consensus_reached", False):
            return "synthesize"

        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "synthesize"

        return "iterate"

    async def _synthesize_node(self, state: AgentState) -> Dict:
        """Synthesize final response from agent outputs."""
        self.logger.info("Synthesizing final response")

        agent_outputs = state.get("agent_outputs", [])

        synthesis_prompt = f"""Query: {state['query']}

Agent Outputs:
"""

        for output in agent_outputs:
            synthesis_prompt += f"""
{output['agent'].upper()} (confidence={output['confidence']:.2f}):
{output['response']}

"""

        synthesis_prompt += """
Synthesize these outputs into a comprehensive final response.
Prioritize higher-confidence outputs. Integrate insights from all agents.

Final Response:"""

        try:
            response = await self.model_adapter.generate(
                prompt=synthesis_prompt,
                temperature=0.5,
            )
            final_response = response.text
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            best_output = max(agent_outputs, key=lambda o: o["confidence"])
            final_response = best_output["response"]

        metadata = {
            "agents_used": [o["agent"] for o in agent_outputs],
            "confidence_scores": state.get("confidence_scores", {}),
            "consensus_score": state.get("consensus_score", 0.0),
            "iterations": state.get("iteration", 0),
            "mcts_config": state.get("mcts_config", {}),
        }

        if state.get("mcts_stats"):
            metadata["mcts_stats"] = state["mcts_stats"]

        return {
            "final_response": final_response,
            "metadata": metadata,
        }


class IntegratedFramework:
    """
    Integrated multi-agent framework with new MCTS core.

    Maintains backward compatibility with original process() signature.
    """

    def __init__(
        self,
        model_adapter,
        logger,
        vector_store=None,
        embedding_model=None,
        hrm_config: Optional[Dict] = None,
        trm_config: Optional[Dict] = None,
        mcts_config: Optional[MCTSConfig] = None,
        top_k_retrieval: int = 5,
        max_iterations: int = 3,
        consensus_threshold: float = 0.75,
        enable_parallel_agents: bool = True,
    ):
        """
        Initialize integrated framework.

        Backward compatible with LangGraphMultiAgentFramework.
        """
        self.model_adapter = model_adapter
        self.logger = logger
        self.vector_store = vector_store

        # Import agents (would be real imports in production)
        try:
            from improved_hrm_agent import HRMAgent
            from improved_trm_agent import TRMAgent

            self.hrm_agent = HRMAgent(
                model_adapter=model_adapter,
                logger=logger,
                **(hrm_config or {}),
            )
            self.trm_agent = TRMAgent(
                model_adapter=model_adapter,
                logger=logger,
                **(trm_config or {}),
            )
        except ImportError:
            self.hrm_agent = None
            self.trm_agent = None
            self.logger.warning("Could not import HRM/TRM agents")

        # Build graph
        self.graph_builder = GraphBuilder(
            hrm_agent=self.hrm_agent,
            trm_agent=self.trm_agent,
            model_adapter=model_adapter,
            logger=logger,
            vector_store=vector_store,
            mcts_config=mcts_config,
            top_k_retrieval=top_k_retrieval,
            max_iterations=max_iterations,
            consensus_threshold=consensus_threshold,
            enable_parallel_agents=enable_parallel_agents,
        )

        # Compile graph
        if StateGraph is not None:
            self.graph = self.graph_builder.build_graph()
            self.memory = MemorySaver() if MemorySaver else None
            self.app = (
                self.graph.compile(checkpointer=self.memory)
                if self.memory
                else self.graph.compile()
            )
        else:
            self.graph = None
            self.app = None

        self.logger.info("Integrated framework initialized with new MCTS core")

    async def process(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Process query through LangGraph.

        Backward compatible with original signature.

        Args:
            query: User query to process
            use_rag: Enable RAG context retrieval
            use_mcts: Enable MCTS simulation
            config: Optional LangGraph config

        Returns:
            Dictionary with response, metadata, and state
        """
        if self.app is None:
            raise RuntimeError("LangGraph not available. Install with: pip install langgraph")

        initial_state = {
            "query": query,
            "use_rag": use_rag,
            "use_mcts": use_mcts,
            "iteration": 0,
            "max_iterations": self.graph_builder.max_iterations,
            "agent_outputs": [],
        }

        config = config or {"configurable": {"thread_id": "default"}}

        result = await self.app.ainvoke(initial_state, config=config)

        return {
            "response": result.get("final_response", ""),
            "metadata": result.get("metadata", {}),
            "state": result,
        }

    def get_experiment_tracker(self) -> ExperimentTracker:
        """Get the experiment tracker for analysis."""
        return self.graph_builder.experiment_tracker

    def set_mcts_seed(self, seed: int) -> None:
        """Set MCTS seed for deterministic behavior."""
        self.graph_builder.mcts_engine.reset_seed(seed)
        self.graph_builder.mcts_config.seed = seed
