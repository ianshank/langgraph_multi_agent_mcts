"""
Tests for Phase 3: Continuous Learning & Optimization.
"""

import pytest
import torch
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.training.system_config import SystemConfig
from src.training.model_registry import ModelRegistry, ModelVersion
from src.training.continuous_learning_manager import ContinuousLearningManager
from src.framework.mcts.llm_guided.integration import (
    HRMAdapter, TRMAdapter, MetaControllerAdapter, UnifiedSearchOrchestrator
)

@pytest.fixture
def temp_dirs():
    """Create temp directories for testing."""
    root = Path(tempfile.mkdtemp())
    data_dir = root / "data"
    registry_dir = root / "registry"
    data_dir.mkdir()
    registry_dir.mkdir()
    yield data_dir, registry_dir
    shutil.rmtree(root)

@pytest.fixture
def mock_orchestrator():
    """Mock DistillationOrchestrator to avoid real training."""
    with patch("src.training.continuous_learning_manager.DistillationOrchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.output_dir = Path("mock_outputs")
        # Mock training methods to return metrics
        instance.distill_hrm.return_value = {"avg_loss": 0.5}
        instance.distill_trm.return_value = {"avg_loss": 0.4}
        instance.distill_meta_controller.return_value = {"avg_loss": 0.3}
        yield instance

class TestModelRegistry:
    def test_register_and_promote(self, temp_dirs):
        data_dir, registry_dir = temp_dirs
        registry = ModelRegistry(registry_dir)
        
        # Create a dummy model file
        dummy_model = data_dir / "model.pt"
        dummy_model.write_text("dummy content")
        
        # Register
        v_id = registry.register_model(
            source_path=dummy_model,
            model_type="hrm",
            metrics={"loss": 0.5},
            tags=["candidate"]
        )
        
        # Verify
        assert v_id in registry.versions
        assert registry.versions[v_id].metrics["loss"] == 0.5
        assert (registry_dir / "hrm" / v_id / "model.pt").exists()
        
        # Promote
        registry.promote_to_best(v_id)
        assert "best" in registry.versions[v_id].tags
        
        # Check get_best
        best = registry.get_best_model_version("hrm")
        assert best.version_id == v_id

class TestContinuousLearningManager:
    def test_trigger_logic(self, temp_dirs, mock_orchestrator):
        data_dir, registry_dir = temp_dirs
        config = SystemConfig()
        
        manager = ContinuousLearningManager(
            config, data_dir, registry_dir,
            min_samples_for_training=5
        )
        manager.orchestrator = mock_orchestrator
        
        # Mock DataCollector statistics
        manager.data_collector = MagicMock()
        manager.data_collector.get_statistics.return_value = {"total_examples": 10}
        
        # Mock file existence for registration
        mock_orchestrator.output_dir = data_dir / "builds"
        mock_orchestrator.output_dir.mkdir()
        (mock_orchestrator.output_dir / "hrm_agent.pt").write_text("hrm")
        (mock_orchestrator.output_dir / "trm_agent.pt").write_text("trm")
        (mock_orchestrator.output_dir / "meta_controller").mkdir() # Directory for meta
        
        # Trigger
        manager.check_and_train()
        
        # Verify training called
        mock_orchestrator.distill_hrm.assert_called()
        mock_orchestrator.distill_trm.assert_called()
        mock_orchestrator.distill_meta_controller.assert_called()
        
        # Verify registration in registry
        assert len(manager.registry.versions) >= 2 # HRM, TRM registered (Meta might fail due to dir structure mock)

class TestHotReloading:
    def test_adapter_hot_reload(self, temp_dirs):
        data_dir, registry_dir = temp_dirs
        
        # Create dummy state dict
        model = torch.nn.Linear(10, 10)
        path = data_dir / "test_model.pt"
        torch.save(model.state_dict(), path)
        
        # Mock creating agent to avoid full config deps
        with patch("src.agents.hrm_agent.create_hrm_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent
            
            adapter = HRMAdapter(encoder=MagicMock())
            adapter.load_model(path)
            
            assert adapter.has_neural_agent
            mock_agent.load_state_dict.assert_called()

    def test_orchestrator_hot_reload_integration(self):
        # Test unified orchestrator calls adapter reload
        hrm = MagicMock()
        trm = MagicMock()
        meta = MagicMock()
        
        orch = UnifiedSearchOrchestrator(
            llm_client=None,
            hrm_adapter=hrm,
            trm_adapter=trm,
            meta_controller_adapter=meta
        )
        
        orch.hot_reload(hrm_path="path/to/hrm", trm_path="path/to/trm")
        
        hrm.load_model.assert_called_with("path/to/hrm", "cpu")
        trm.load_model.assert_called_with("path/to/trm", "cpu")
        meta.load_model.assert_not_called()
