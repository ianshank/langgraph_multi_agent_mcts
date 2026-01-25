"""
Benchmark Latency: LLM vs Neural Pipeline.

This script measures the inference time of the UnifiedSearchOrchestrator
in two modes:
1. LLM-only (Simulated delay)
2. Neural (Local Encoder-Agent-Decoder)

Usage:
    python scripts/benchmark_latency.py
"""

import asyncio
import time
import torch
from unittest.mock import MagicMock, AsyncMock

from src.framework.mcts.llm_guided.integration import (
    UnifiedSearchOrchestrator,
    IntegrationConfig,
    HRMAdapter, 
    TRMAdapter
)
from src.training.system_config import SystemConfig
from src.framework.mcts.llm_guided.config import LLMGuidedMCTSConfig, LLMGuidedMCTSPreset, create_llm_mcts_preset
from src.agents.common.system_encoder import SystemEncoder
from src.agents.common.text_decoder import SystemDecoder
from src.agents.hrm_agent import HRMAgent, HRMConfig
from src.agents.trm_agent import TRMAgent, TRMConfig

async def benchmark():
    print("=" * 60)
    print("BENCHMARK: LLM vs Neural Infrastructure Latency")
    print("=" * 60)

    # 1. Setup Common Mocks
    llm_client = AsyncMock()
    # Simulate LLM Network Latency (e.g. 500ms)
    async def fast_llm_response(*args, **kwargs):
        await asyncio.sleep(0.5) 
        return "LLM Response"
    llm_client.completion = fast_llm_response
    
    # Mock MCTS Engine to be instant (focus on adaptation overhead)
    mcts_engine = AsyncMock()
    mcts_engine.search.return_value = MagicMock(
        solution_found=True, best_code="pass", best_value=1.0, 
        num_iterations=1, num_expansions=1, llm_calls=0, tokens_used=0
    )
    mcts_engine._data_collector = MagicMock()
    
    meta_controller = MagicMock()
    meta_controller.route.return_value = MagicMock(selected_agent="LLM_MCTS", confidence=1.0)

    # ---------------------------------------------------------
    # Scenario A: LLM Fallback (No Neural Agents)
    # ---------------------------------------------------------
    config_llm = IntegrationConfig(
        use_hrm_decomposition=True, # Will fallback to LLM
        use_trm_refinement=True      # Will fallback to LLM
    )
    
    hrm_adapter_llm = HRMAdapter(None, llm_client, config_llm)
    trm_adapter_llm = TRMAdapter(None, llm_client, config_llm) # No neural agent
    
    # Use Fast preset for benchmark
    mcts_config = create_llm_mcts_preset(LLMGuidedMCTSPreset.FAST)

    orch_llm = UnifiedSearchOrchestrator(
        llm_client, mcts_config, config_llm, 
        hrm_adapter=hrm_adapter_llm, trm_adapter=trm_adapter_llm,
        meta_controller_adapter=meta_controller
    )
    orch_llm._mcts_engine = mcts_engine
    
    # Warmup
    # We need to patch the adapters' internal _decompose_llm to actually be called
    # But wait, logic in adapters checks 'if self.has_neural_agent'.
    # If no neural agent, it calls `_decompose_llm` or `_refine_llm`.
    
    print("\n[Scenario A] Running LLM Fallback mode (Simulating 500ms network lag)...")
    start = time.perf_counter()
    # We call internal methods to isolate component latency if possible, 
    # but let's run a full search flow where we force decomposition.
    # The default route is LLM_MCTS, so let's force HRM via config or context?
    # Integration logic: if config.use_hrm_decomposition is True, it tries to decompose.
    # But it also checks routing decision.
    
    # For benchmark simplicity, we'll benchmark the ADAPTER methods directly.
    # This isolates the component we optimized.
    
    await hrm_adapter_llm.decompose("Test Problem")
    end = time.perf_counter()
    llm_latency = (end - start) * 1000
    print(f"  Latency: {llm_latency:.2f} ms")


    # ---------------------------------------------------------
    # Scenario B: Neural Pipeline (Local Models)
    # ---------------------------------------------------------
    print("\n[Scenario B] Running Neural Distillation mode (Local CPU Inference)...")
    
    # Use real (small) models or mocks?
    # To be realistic, let's use the real classes but tiny configs.
    # Note: If no transformers installed, this will fail or we mock.
    # SystemEncoder/Decoder require transformers.
    
    try:
        encoder = SystemEncoder(model_name="prajjwal1/bert-mini", device="cpu")
        decoder = SystemDecoder(model_name="distilgpt2", device="cpu", latent_dim=256)
        
        hrm_config = HRMConfig(h_dim=256, l_dim=128, num_h_layers=1)
        hrm_agent = HRMAgent(hrm_config, device="cpu")
        
        # We need to project encoder output (256) to match if needed, but BERT-mini is 256.
        # Check encoder size
        if encoder.hidden_size != 256:
            print(f"  Warning: Encoder hidden size {encoder.hidden_size} != Agent 256. Mismatch might occur.")
            
    except Exception as e:
        print(f"  Skipping real model benchmark due to missing deps: {e}")
        print("  Using Mocks for Neural Bench...")
        encoder = MagicMock()
        encoder.return_value = torch.randn(1, 10, 256)
        encoder.hidden_size = 256 # Ensure hidden size is set on mock
        decoder = MagicMock()
        decoder.generate.return_value = ["Neural Plan"]
        hrm_agent = MagicMock()
        hrm_agent.return_value = MagicMock(final_state=torch.randn(1, 1, 256))

    config_neural = IntegrationConfig(
        use_hrm_decomposition=True,
        distillation_mode=False
    )
    
    hrm_adapter_neural = HRMAdapter(
        hrm_agent=hrm_agent, 
        llm_client=llm_client, 
        config=config_neural,
        encoder=encoder,
        decoder=decoder
    )
    
    # Warmup (PyTorch overhead)
    await hrm_adapter_neural.decompose("Warmup")
    
    start = time.perf_counter()
    await hrm_adapter_neural.decompose("Test Problem")
    end = time.perf_counter()
    neural_latency = (end - start) * 1000
    print(f"  Latency: {neural_latency:.2f} ms")
    
    print("-" * 60)
    print(f"Speedup: {llm_latency / (neural_latency + 1e-6):.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(benchmark())
