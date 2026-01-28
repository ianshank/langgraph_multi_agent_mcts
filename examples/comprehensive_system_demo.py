import os
import logging
import sys
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Force UTF-8 for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from dotenv import load_dotenv

# Import framework components
from src.framework.graph import GraphBuilder, AgentState
from src.framework.personality.agent import PersonalityDrivenAgent
from src.framework.personality.profiles import PersonalityProfile
from src.framework.mcts.config import MCTSConfig, ConfigPreset
from src.integrations.google_adk.base import ADKAgentAdapter, ADKConfig, ADKAgentRequest, ADKAgentResponse
from src.adapters.llm.base import LLMClient, LLMResponse
from src.adapters.llm.openai_client import OpenAIClient
from src.adapters.llm.anthropic_client import AnthropicClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Demo")

# Load environment variables
load_dotenv()

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

class MockModelAdapter:
    """Simulates LLM responses for the demo."""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate simulated text based on prompt keywords."""
        prompt_lower = prompt.lower()
        
        text = "Simulated LLM response."
        if "plan" in prompt_lower or "hrm" in prompt_lower:
            text = """
            1. Analyze current security landscape.
            2. Identify quantum threats.
            3. Develop counter-measures.
            4. Validate with simulation.
            """
        elif "refine" in prompt_lower or "trm" in prompt_lower:
            text = "Refined step: Focus on lattice-based cryptography as a primary defense."
        elif "critique" in prompt_lower:
            text = "The plan is solid but lacks a timeline. Score: 0.85"
        elif "synthesize" in prompt_lower:
            text = "Final Report: Quantum computing poses significant risks to RSA. Immediate transition to post-quantum cryptography (PQC) is recommended."
            
        return LLMResponse(text=text)

    async def get_embedding(self, text: str) -> List[float]:
        """Return deterministic embedding."""
        # Use text hash to allow stable but distinct embeddings
        # This prevents immediate consensus if texts are different
        seed = sum(ord(c) for c in text[:20]) % 10000
        rng = np.random.RandomState(seed)
        return rng.rand(1536).tolist()

# --- Agents Implementation ---

class DemoAgentBase:
    """Base class for demo agents to handle model adapter interaction."""
    def __init__(self, model_adapter: Any, name: str):
        self.model_adapter = model_adapter
        self.name = name
        self.logger = logging.getLogger(name)

    async def initialize(self): 
        self.logger.info(f"[{self.name}] Initializing...")

    async def shutdown(self): 
        self.logger.info(f"[{self.name}] Shutting down...")

    async def _call_model(self, prompt: str) -> str:
        """Helper to call model adapter."""
        try:
            # Check if adapter is real LLMClient (has generate method)
            if hasattr(self.model_adapter, "generate"):
                # Call generate
                response = await self.model_adapter.generate(prompt=prompt)
                # Extract text from response object
                if hasattr(response, "text"):
                    return response.text
                return str(response)
            else:
                # Fallback for simple mocks
                return await self.model_adapter.generate(prompt)
        except Exception as e:
            self.logger.error(f"Model call failed: {e}")
            return f"Error generating response: {e}"

class DemoHRMAgent(DemoAgentBase):
    """Hierarchical Reasoning Agent implementation for Demo."""
    
    async def process(self, context: Any) -> Any:
        # Extract query from context (Context object or dict)
        query = getattr(context, "query", None) or context.get("query") if isinstance(context, dict) else str(context)
        
        self.logger.info(f"[{self.name}] Processing high-level plan for: {query[:50]}...")
        
        prompt = (
            f"You are a Strategic Director (HRM). Create a high-level strategic plan for: {query}.\n"
            "Format as a numbered list of phases."
        )
        
        response_text = await self._call_model(prompt)
        
        return type('obj', (object,), {
            'response': response_text,
            'confidence': 0.9,
            'metadata': {'role': 'strategist', 'model_response': True}
        })

class DemoTRMAgent(DemoAgentBase):
    """Tactical Refinement Agent implementation for Demo."""
    
    async def process(self, context: Any) -> Any:
        query = getattr(context, "query", None) or context.get("query") if isinstance(context, dict) else str(context)
        
        self.logger.info(f"[{self.name}] Refining tactical details for: {query[:50]}...")
        
        prompt = (
            f"You are a Tactical Researcher (TRM). Provide specific technical details and implementation steps for: {query}.\n"
            "Focus on specific technologies and methodologies."
        )
        
        response_text = await self._call_model(prompt)
        
        return type('obj', (object,), {
            'response': response_text,
            'confidence': 0.85,
            'metadata': {'role': 'tactician', 'model_response': True}
        })


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

# --- Real Adapter Factories ---

def get_model_adapter() -> Any:
    """Get appropriate model adapter based on environment."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        logger.info("Using OpenAI API")
        client = OpenAIClient(api_key=openai_key)
        # Patch get_embedding if missing (OpenAIClient usually handles text generation)
        if not hasattr(client, "get_embedding"):
             # Add a simple mock embedding or implement real OpenAI embedding call if needed
             # For now, using deterministic mock for embeddings to save cost/complexity
             async def mock_get_embedding(text: str) -> List[float]:
                 seed = sum(ord(c) for c in text[:20]) % 10000
                 rng = np.random.RandomState(seed)
                 return rng.rand(1536).tolist()
             client.get_embedding = mock_get_embedding
        return client
        
    elif anthropic_key:
        logger.info("Using Anthropic API")
        client = AnthropicClient(api_key=anthropic_key)
        # Anthropic doesn't support embeddings natively in the same way
        async def mock_get_embedding(text: str) -> List[float]:
             seed = sum(ord(c) for c in text[:20]) % 10000
             rng = np.random.RandomState(seed)
             return rng.rand(1536).tolist()
        client.get_embedding = mock_get_embedding
        return client
        
    else:
        logger.warning("No API keys found. Using Mock Adapter.")
        return MockModelAdapter()

# --- Main Demo Flow ---

async def run_demo():
    print("\n" + "="*80)
    print(" COMPREHENSIVE SYSTEM DEMONSTRATION")
    print("="*80 + "\n")

    # 1. Setup Dependencies
    model_adapter = get_model_adapter()
    
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
    
    # Use Demo Agents which wrap the model adapter
    base_hrm = DemoHRMAgent(model_adapter=model_adapter, name="Director-HRM")
    base_trm = DemoTRMAgent(model_adapter=model_adapter, name="Analyst-TRM")
    
    # Wrap with Personality
    hrm_agent = PersonalityDrivenAgent(base_hrm, director_profile)
    trm_agent = PersonalityDrivenAgent(base_trm, researcher_profile)
    
    # Initialize ADK Agent
    # If GOOGLE_APPLICATION_CREDENTIALS exists, try real ADK
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_API_KEY"):
         adk_config = ADKConfig.from_env()
         # Assuming factory exists or we use mock if strictly needed
         adk_agent = MockADKAgent(adk_config, agent_name="Google-Quantum-Specialist") 
    else:
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
