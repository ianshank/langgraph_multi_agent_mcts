"""
Comprehensive Demonstration of LangGraph Multi-Agent MCTS System.

This script demonstrates the full capabilities of the application, including:
1.  Multi-Agent Collaboration (HRM & TRM Agents)
2.  Personality-Driven Behavior (High Performer vs. Explorer profiles)
3.  Monte Carlo Tree Search (MCTS) for decision making
4.  Google ADK Agent Integration (Simulated)
5.  LangGraph State Management

The demo runs with a simulated LLM adapter to require no API keys.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

# Force UTF-8 for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from langchain_core.language_models.fake import FakeListLLM

# Import framework components
from src.framework.graph import GraphBuilder, AgentState
from src.framework.personality.agent import PersonalityDrivenAgent
from src.framework.personality.profiles import PersonalityProfile
from src.framework.mcts.config import MCTSConfig, ConfigPreset
from src.integrations.google_adk.base import ADKAgentAdapter, ADKConfig, ADKAgentRequest, ADKAgentResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Demo")

# --- Custom Graph Builder to Fix Iteration Logic ---

class DemoGraphBuilder(GraphBuilder):
    """Custom GraphBuilder that properly increments iteration count."""
    
    def _evaluate_consensus_node(self, state: AgentState) -> dict:
        """Override to increment iteration counter."""
        # Call parent implementation
        result = super()._evaluate_consensus_node(state)
        
        # Increment iteration
        current_iteration = state.get("iteration", 0)
        result["iteration"] = current_iteration + 1
        
        logger.info(f"Evaluation complete. Moving to iteration {result['iteration']}")
        return result

# --- Mocks for Independence ---

class MockResponse:
    def __init__(self, text):
        self.text = text

class MockModelAdapter:
    """Simulates LLM responses for the demo."""
    
    async def generate(self, prompt: str, **kwargs) -> MockResponse:
        """Generate simulated text based on prompt keywords."""
        prompt_lower = prompt.lower()
        
        if "plan" in prompt_lower or "hrm" in prompt_lower:
            return MockResponse("""
            1. Analyze current security landscape.
            2. Identify quantum threats.
            3. Develop counter-measures.
            4. Validate with simulation.
            """)
        elif "refine" in prompt_lower or "trm" in prompt_lower:
            return MockResponse("Refined step: Focus on lattice-based cryptography as a primary defense.")
        elif "critique" in prompt_lower:
            return MockResponse("The plan is solid but lacks a timeline. Score: 0.85")
        elif "synthesize" in prompt_lower:
            return MockResponse("Final Report: Quantum computing poses significant risks to RSA. Immediate transition to post-quantum cryptography (PQC) is recommended.")
        else:
            return MockResponse("Simulated LLM response.")

    async def get_embedding(self, text: str) -> List[float]:
        """Return deterministic embedding."""
        # Use text hash to allow stable but distinct embeddings
        # This prevents immediate consensus if texts are different
        seed = sum(ord(c) for c in text[:20]) % 10000
        rng = np.random.RandomState(seed)
        return rng.rand(1536).tolist()

class MockHRMAgent:
    """Simulates Hierarchical Reasoning Agent."""
    def __init__(self, name="HRM"):
        self.name = name
    
    async def process(self, context: Any) -> Any:
        logger.info(f"[{self.name}] Processing high-level plan...")
        await asyncio.sleep(0.5)
        return type('obj', (object,), {
            'response': "Strategic Plan: 1. Assessment, 2. Mitigation, 3. Deployment",
            'confidence': 0.9,
            'metadata': {'role': 'strategist'}
        })

    async def initialize(self): pass
    async def shutdown(self): pass

class MockTRMAgent:
    """Simulates Tactical Refinement Agent."""
    def __init__(self, name="TRM"):
        self.name = name
    
    async def process(self, context: Any) -> Any:
        logger.info(f"[{self.name}] Refining tactical details...")
        await asyncio.sleep(0.5)
        return type('obj', (object,), {
            'response': "Tactical Detail: Use Kyber-1024 for key encapsulation.",
            'confidence': 0.85,
            'metadata': {'role': 'tactician'}
        })

    async def initialize(self): pass
    async def shutdown(self): pass

class MockADKAgent(ADKAgentAdapter):
    """Simulates a Google ADK Agent (e.g., a specialist tool)."""
    async def _agent_initialize(self) -> None:
        logger.info(f"[{self.agent_name}] Initializing Google ADK connection...")
    
    async def _agent_invoke(self, request: ADKAgentRequest) -> ADKAgentResponse:
        logger.info(f"[{self.agent_name}] Analyzing with specialized model: {request.query}")
        return ADKAgentResponse(
            result="ADK Analysis: Quantum drift detected in sector 7G.",
            metadata={"backend": "vertex_ai_simulated"},
            status="success"
        )

# --- Main Demo Flow ---

async def run_demo():
    print("\n" + "="*80)
    print(" COMPREHENSIVE SYSTEM DEMONSTRATION")
    print("="*80 + "\n")

    # 1. Setup Dependencies
    model_adapter = MockModelAdapter()
    
    # 2. Configure Personalities
    print(" Configuring Personality Profiles...")
    
    # Director Profile (High Loyalty/Aspiration) -> HRM
    director_profile = PersonalityProfile.high_performer()
    print(f"  - Director Profile: {director_profile.trait_vector} (Loyalty/Curiosity/Aspiration/Ethical/Transparency)")
    
    # Researcher Profile (High Curiosity) -> TRM
    researcher_profile = PersonalityProfile.explorer()
    print(f"  - Researcher Profile: {researcher_profile.trait_vector}")

    # 3. Initialize Agents
    print("\n Initializing Agents...")
    
    base_hrm = MockHRMAgent(name="Director-HRM")
    base_trm = MockTRMAgent(name="Analyst-TRM")
    
    # Wrap with Personality
    hrm_agent = PersonalityDrivenAgent(base_hrm, director_profile)
    trm_agent = PersonalityDrivenAgent(base_trm, researcher_profile)
    
    # Initialize ADK Agent
    adk_config = ADKConfig(project_id="demo-project", backend="local")
    adk_agent = MockADKAgent(adk_config, agent_name="Google-Quantum-Specialist")

    # 4. Build Integrated Graph
    print("\n  Building LangGraph with MCTS...")
    graph_builder = DemoGraphBuilder(
        hrm_agent=hrm_agent,
        trm_agent=trm_agent,
        model_adapter=model_adapter,
        logger=logger,
        mcts_config=MCTSConfig(max_rollout_depth=5),
        adk_agents={"quantum_tool": adk_agent}
    )
    
    app = graph_builder.build_graph().compile()
    
    # 5. Execute Scenario
    query = "Develop a defense strategy against quantum decryption threats for 2026."
    print(f"\n Processing Query: '{query}'\n")
    print("-" * 60)

    initial_state = {
        "query": query,
        "use_mcts": True,
        "use_rag": True,
        "iteration": 0,
        "max_iterations": 2,
        "agent_outputs": []
    }

    # Run the graph
    async for event in app.astream(initial_state):
        for node, state in event.items():
            print(f"\n Node Completed: {node}")
            
            # Inspect Agent Outputs if present
            if node == "hrm_agent":
                print(f"    Director (HRM): Strategic assessment complete.")
                # Show personality report for HRM
                print(f"    Personality Influence: {hrm_agent._decision_history[-1].explanation if hrm_agent._decision_history else 'N/A'}")
            
            elif node == "trm_agent":
                print(f"    Researcher (TRM): Tactical refinement complete.")
                print(f"    Personality Influence: {trm_agent._decision_history[-1].explanation if trm_agent._decision_history else 'N/A'}")
            
            elif node == "mcts_simulator":
                print(f"    MCTS: Simulation ran. Optimized path identified.")
            
            elif node == "synthesize":
                print(f"\n Final Synthesis: {state.get('final_response', 'Processing complete')}")

    # 6. Final Reports
    print("\n" + "="*80)
    print(" POST-EXECUTION REPORTS")
    print("="*80)
    
    print("\nHRM Agent (Director) Personality Report:")
    print(hrm_agent.generate_personality_report())
    
    print("\nTRM Agent (Researcher) Personality Report:")
    print(trm_agent.generate_personality_report())

if __name__ == "__main__":
    asyncio.run(run_demo())

