"""
Comprehensive tests for extended dataset loaders.

Tests all new dataset loaders (ARC, GSM8K, IDoFT, HumanEval, Chess, BIG-Bench Hard)
following pytest best practices of 2025.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.dataset_loader import (
    ARCLoader,
    BIGBenchHardLoader,
    ChessGamesLoader,
    CombinedDatasetLoader,
    DatasetSample,
    DatasetStatistics,
    GSM8KLoader,
    HumanEvalLoader,
    IDoFTLoader,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_cache_dir():
    """Provide a temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_arc_dataset():
    """Mock ARC dataset structure."""
    return {
        "train": [
            {
                "train": {
                    "input": [[1, 2], [3, 4]],
                    "output": [[5, 6], [7, 8]],
                },
                "test": {
                    "input": [[9, 10]],
                    "output": [[11, 12]],
                },
            }
        ]
    }


@pytest.fixture
def mock_gsm8k_dataset():
    """Mock GSM8K dataset structure."""
    return {
        "train": [
            {
                "question": "Janet has 3 apples. She buys 2 more. How many does she have?",
                "answer": "She starts with 3 apples.\nShe buys 2 more.\n3 + 2 = 5\n#### 5",
            },
            {
                "question": "A car travels 60 miles in 1 hour. How far in 3 hours?",
                "answer": "60 miles per hour.\n3 hours total.\n60 * 3 = 180\n#### 180",
            },
        ],
        "test": [
            {
                "question": "Test question?",
                "answer": "Test answer.\n#### 42",
            }
        ],
    }


@pytest.fixture
def mock_humaneval_dataset():
    """Mock HumanEval dataset structure."""
    return {
        "test": [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n    ",
                "canonical_solution": "return a + b",
                "test": "assert add(1, 2) == 3",
                "entry_point": "add",
            }
        ]
    }


@pytest.fixture
def mock_chess_dataset():
    """Mock Chess dataset structure."""
    return {
        "train": [
            {
                "date": "2024.01.01",
                "white_elo": 2400,
                "black_elo": 2350,
                "end_type": "checkmate",
                "moves_uci": "e2e4 e7e5 g1f3",
                "moves_san": "e4 e5 Nf3",
            },
            {
                "date": "2024.01.02",
                "white_elo": 1800,  # Below min_elo threshold
                "black_elo": 1750,
                "end_type": "resignation",
                "moves_uci": "d2d4",
                "moves_san": "d4",
            },
        ]
    }


@pytest.fixture
def mock_bbh_dataset():
    """Mock BIG-Bench Hard dataset structure."""
    return {
        "train": [
            {
                "task": "causal_judgement",
                "input": "What caused the event?",
                "target": "The trigger",
            },
            {
                "task": "logical_deduction",
                "input": "Who is taller?",
                "target": "Person A",
            },
        ]
    }


@pytest.fixture
def idoft_test_data(temp_cache_dir):
    """Create mock IDoFT data files."""
    data_dir = Path(temp_cache_dir) / "idoft_data"
    data_dir.mkdir()

    # Create sample flaky test data
    test_data = [
        {
            "project": "test-project-1",
            "test_name": "testAsyncTimeout",
            "category": "async_wait",
            "test_code": "public void testAsyncTimeout() { /* code */ }",
            "failure_rate": 0.15,
        },
        {
            "project": "test-project-2",
            "test_name": "testConcurrency",
            "category": "concurrency",
            "test_code": "public void testConcurrency() { /* code */ }",
            "failure_rate": 0.22,
        },
    ]

    test_file = data_dir / "flaky_tests.json"
    with open(test_file, "w") as f:
        json.dump(test_data, f)

    return str(data_dir)


# =============================================================================
# ARCLoader Tests
# =============================================================================


class TestARCLoader:
    """Test suite for ARCLoader."""

    def test_initialization(self, temp_cache_dir):
        """Test ARCLoader initialization."""
        loader = ARCLoader(cache_dir=temp_cache_dir)
        assert loader.cache_dir == temp_cache_dir
        assert loader._loaded_samples == []

    @patch("src.data.dataset_loader.load_dataset")
    def test_load_arc_dataset(self, mock_load_dataset, temp_cache_dir, mock_arc_dataset):
        """Test loading ARC dataset."""
        mock_load_dataset.return_value = mock_arc_dataset

        loader = ARCLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="train")

        assert len(samples) == 1
        assert isinstance(samples[0], DatasetSample)
        assert samples[0].domain == "abstract_reasoning"
        assert samples[0].difficulty == "hard"
        assert "task_data" in samples[0].metadata

    @patch("src.data.dataset_loader.load_dataset")
    def test_arc_statistics(self, mock_load_dataset, temp_cache_dir, mock_arc_dataset):
        """Test ARC statistics generation."""
        mock_load_dataset.return_value = mock_arc_dataset

        loader = ARCLoader(cache_dir=temp_cache_dir)
        loader.load(split="train")
        stats = loader.get_statistics()

        assert isinstance(stats, DatasetStatistics)
        assert stats.total_samples == 1
        assert "abstract_reasoning" in stats.domains

    @patch("src.data.dataset_loader.load_dataset")
    def test_arc_batch_iteration(self, mock_load_dataset, temp_cache_dir, mock_arc_dataset):
        """Test ARC batch iteration."""
        mock_load_dataset.return_value = mock_arc_dataset

        loader = ARCLoader(cache_dir=temp_cache_dir)
        loader.load(split="train")

        batches = list(loader.iterate_samples(batch_size=1))
        assert len(batches) == 1
        assert len(batches[0]) == 1


# =============================================================================
# GSM8KLoader Tests
# =============================================================================


class TestGSM8KLoader:
    """Test suite for GSM8KLoader."""

    def test_initialization(self, temp_cache_dir):
        """Test GSM8KLoader initialization."""
        loader = GSM8KLoader(cache_dir=temp_cache_dir)
        assert loader.cache_dir == temp_cache_dir
        assert loader._loaded_samples == []

    @patch("src.data.dataset_loader.load_dataset")
    def test_load_gsm8k_dataset(self, mock_load_dataset, temp_cache_dir, mock_gsm8k_dataset):
        """Test loading GSM8K dataset."""
        mock_load_dataset.return_value = mock_gsm8k_dataset

        loader = GSM8KLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="train", config="main")

        assert len(samples) == 2
        assert all(isinstance(s, DatasetSample) for s in samples)
        assert all(s.domain == "mathematics" for s in samples)
        assert all(s.difficulty == "medium" for s in samples)

    @patch("src.data.dataset_loader.load_dataset")
    def test_gsm8k_reasoning_steps(self, mock_load_dataset, temp_cache_dir, mock_gsm8k_dataset):
        """Test GSM8K reasoning step extraction."""
        mock_load_dataset.return_value = mock_gsm8k_dataset

        loader = GSM8KLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="train")

        # Check reasoning steps are extracted
        for sample in samples:
            assert sample.reasoning_steps is not None
            assert len(sample.reasoning_steps) > 0

    @patch("src.data.dataset_loader.load_dataset")
    def test_get_reasoning_samples(self, mock_load_dataset, temp_cache_dir, mock_gsm8k_dataset):
        """Test filtering reasoning samples."""
        mock_load_dataset.return_value = mock_gsm8k_dataset

        loader = GSM8KLoader(cache_dir=temp_cache_dir)
        loader.load(split="train")
        reasoning_samples = loader.get_reasoning_samples()

        assert len(reasoning_samples) == 2  # Both samples have multi-step reasoning


# =============================================================================
# IDoFTLoader Tests
# =============================================================================


class TestIDoFTLoader:
    """Test suite for IDoFTLoader."""

    def test_initialization(self, temp_cache_dir, idoft_test_data):
        """Test IDoFTLoader initialization."""
        loader = IDoFTLoader(cache_dir=temp_cache_dir, data_path=idoft_test_data)
        assert loader.cache_dir == temp_cache_dir
        assert loader.data_path == idoft_test_data

    def test_load_from_local(self, temp_cache_dir, idoft_test_data):
        """Test loading IDoFT from local files."""
        loader = IDoFTLoader(cache_dir=temp_cache_dir, data_path=idoft_test_data)
        samples = loader.load(split="train")

        assert len(samples) == 2
        assert all(isinstance(s, DatasetSample) for s in samples)
        assert all(s.domain == "quality_engineering" for s in samples)

    def test_filter_by_category(self, temp_cache_dir, idoft_test_data):
        """Test filtering IDoFT by category."""
        loader = IDoFTLoader(cache_dir=temp_cache_dir, data_path=idoft_test_data)
        loader.load(split="train")

        async_samples = loader.get_by_category("async_wait")
        concurrency_samples = loader.get_by_category("concurrency")

        assert len(async_samples) == 1
        assert len(concurrency_samples) == 1
        assert async_samples[0].labels == ["async_wait"]
        assert concurrency_samples[0].labels == ["concurrency"]

    def test_load_without_data_path_raises_error(self, temp_cache_dir):
        """Test that loading without data_path raises appropriate error."""
        loader = IDoFTLoader(cache_dir=temp_cache_dir, data_path=None)

        with pytest.raises(FileNotFoundError):
            loader.load(split="train")

    def test_category_filtering(self, temp_cache_dir, idoft_test_data):
        """Test loading with category filtering."""
        loader = IDoFTLoader(cache_dir=temp_cache_dir, data_path=idoft_test_data)
        samples = loader.load(split="train", categories=["async_wait"])

        assert len(samples) == 1
        assert samples[0].labels == ["async_wait"]


# =============================================================================
# HumanEvalLoader Tests
# =============================================================================


class TestHumanEvalLoader:
    """Test suite for HumanEvalLoader."""

    @patch("src.data.dataset_loader.load_dataset")
    def test_load_humaneval_dataset(self, mock_load_dataset, temp_cache_dir, mock_humaneval_dataset):
        """Test loading HumanEval dataset."""
        mock_load_dataset.return_value = mock_humaneval_dataset

        loader = HumanEvalLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="test")

        assert len(samples) == 1
        assert samples[0].domain == "code_generation"
        assert samples[0].difficulty == "medium"
        assert "task_id" in samples[0].metadata
        assert "canonical_solution" in samples[0].metadata

    @patch("src.data.dataset_loader.load_dataset")
    def test_humaneval_metadata(self, mock_load_dataset, temp_cache_dir, mock_humaneval_dataset):
        """Test HumanEval metadata extraction."""
        mock_load_dataset.return_value = mock_humaneval_dataset

        loader = HumanEvalLoader(cache_dir=temp_cache_dir)
        samples = loader.load()

        sample = samples[0]
        assert sample.metadata["task_id"] == "HumanEval/0"
        assert "def add" in sample.metadata["prompt"]
        assert sample.metadata["entry_point"] == "add"


# =============================================================================
# ChessGamesLoader Tests
# =============================================================================


class TestChessGamesLoader:
    """Test suite for ChessGamesLoader."""

    @patch("src.data.dataset_loader.load_dataset")
    def test_load_chess_games(self, mock_load_dataset, temp_cache_dir, mock_chess_dataset):
        """Test loading Chess games dataset."""
        # Mock dataset as iterable
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter(mock_chess_dataset["train"])
        mock_load_dataset.return_value = {"train": mock_dataset}

        loader = ChessGamesLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="train", min_elo=2000, max_samples=10, streaming=True)

        # Only high ELO game should be included
        assert len(samples) == 1
        assert samples[0].domain == "strategic_planning"

    @patch("src.data.dataset_loader.load_dataset")
    def test_chess_elo_filtering(self, mock_load_dataset, temp_cache_dir, mock_chess_dataset):
        """Test ELO rating filtering."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter(mock_chess_dataset["train"])
        mock_load_dataset.return_value = {"train": mock_dataset}

        loader = ChessGamesLoader(cache_dir=temp_cache_dir)
        samples = loader.load(min_elo=2000, streaming=True)

        # Should only include games with both players >= 2000 ELO
        for sample in samples:
            assert sample.metadata["white_elo"] >= 2000
            assert sample.metadata["black_elo"] >= 2000

    @patch("src.data.dataset_loader.load_dataset")
    def test_get_high_level_games(self, mock_load_dataset, temp_cache_dir, mock_chess_dataset):
        """Test filtering high-level games."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter(mock_chess_dataset["train"])
        mock_load_dataset.return_value = {"train": mock_dataset}

        loader = ChessGamesLoader(cache_dir=temp_cache_dir)
        loader.load(min_elo=2000, streaming=True)

        high_level = loader.get_high_level_games(min_elo=2500)
        # No games meet this threshold in our mock data
        assert len(high_level) == 0


# =============================================================================
# BIGBenchHardLoader Tests
# =============================================================================


class TestBIGBenchHardLoader:
    """Test suite for BIGBenchHardLoader."""

    @patch("src.data.dataset_loader.load_dataset")
    def test_load_bbh_dataset(self, mock_load_dataset, temp_cache_dir, mock_bbh_dataset):
        """Test loading BIG-Bench Hard dataset."""
        mock_load_dataset.return_value = mock_bbh_dataset

        loader = BIGBenchHardLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="train")

        assert len(samples) == 2
        assert all(s.domain == "complex_reasoning" for s in samples)
        assert all(s.difficulty == "hard" for s in samples)

    @patch("src.data.dataset_loader.load_dataset")
    def test_bbh_task_filtering(self, mock_load_dataset, temp_cache_dir, mock_bbh_dataset):
        """Test filtering by specific tasks."""
        mock_load_dataset.return_value = mock_bbh_dataset

        loader = BIGBenchHardLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="train", tasks=["causal_judgement"])

        assert len(samples) == 1
        assert samples[0].labels == ["causal_judgement"]

    @patch("src.data.dataset_loader.load_dataset")
    def test_get_by_task(self, mock_load_dataset, temp_cache_dir, mock_bbh_dataset):
        """Test getting samples by task name."""
        mock_load_dataset.return_value = mock_bbh_dataset

        loader = BIGBenchHardLoader(cache_dir=temp_cache_dir)
        loader.load(split="train")

        causal_samples = loader.get_by_task("causal_judgement")
        logical_samples = loader.get_by_task("logical_deduction")

        assert len(causal_samples) == 1
        assert len(logical_samples) == 1


# =============================================================================
# CombinedDatasetLoader Tests
# =============================================================================


class TestCombinedDatasetLoader:
    """Test suite for CombinedDatasetLoader."""

    def test_initialization(self, temp_cache_dir, idoft_test_data):
        """Test CombinedDatasetLoader initialization."""
        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir, idoft_data_path=idoft_test_data)

        assert loader.cache_dir == temp_cache_dir
        assert hasattr(loader, "arc_loader")
        assert hasattr(loader, "gsm8k_loader")
        assert hasattr(loader, "idoft_loader")
        assert hasattr(loader, "humaneval_loader")
        assert hasattr(loader, "chess_loader")
        assert hasattr(loader, "bbh_loader")

    @patch("src.data.dataset_loader.ARCLoader.load")
    @patch("src.data.dataset_loader.GSM8KLoader.load")
    @patch("src.data.dataset_loader.DABStepLoader.load")
    @patch("src.data.dataset_loader.PRIMUSLoader.load_seed")
    def test_load_all_with_new_datasets(
        self, mock_primus, mock_dabstep, mock_gsm8k, mock_arc, temp_cache_dir
    ):
        """Test loading multiple datasets together."""
        # Mock return values
        mock_dabstep.return_value = []
        mock_primus.return_value = []
        mock_arc.return_value = [DatasetSample(id="arc_1", text="ARC sample", domain="abstract_reasoning")]
        mock_gsm8k.return_value = [
            DatasetSample(id="gsm8k_1", text="Math problem", domain="mathematics")
        ]

        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir)
        samples = loader.load_all(
            dabstep_split="train",
            primus_max_samples=0,
            include_instruct=False,
            include_arc=True,
            include_gsm8k=True,
        )

        # Should have samples from ARC and GSM8K
        assert len(samples) == 2

    def test_get_hrm_training_samples(self, temp_cache_dir):
        """Test getting HRM-specific training samples."""
        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir)
        loader._all_samples = [
            DatasetSample(id="1", text="test", domain="abstract_reasoning"),
            DatasetSample(id="2", text="test", domain="mathematics"),
            DatasetSample(id="3", text="test", domain="code_generation"),
        ]

        hrm_samples = loader.get_hrm_training_samples()
        assert len(hrm_samples) == 2  # abstract_reasoning and mathematics

    def test_get_trm_training_samples(self, temp_cache_dir):
        """Test getting TRM-specific training samples."""
        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir)
        loader._all_samples = [
            DatasetSample(id="1", text="test", metadata={"source": "GSM8K"}),
            DatasetSample(id="2", text="test", metadata={"source": "HumanEval"}),
            DatasetSample(id="3", text="test", metadata={"source": "Other"}),
        ]

        trm_samples = loader.get_trm_training_samples()
        assert len(trm_samples) == 2  # GSM8K and HumanEval

    def test_get_mcts_training_samples(self, temp_cache_dir):
        """Test getting MCTS-specific training samples."""
        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir)
        loader._all_samples = [
            DatasetSample(id="1", text="test", domain="strategic_planning"),
            DatasetSample(id="2", text="test", domain="mathematics"),
        ]

        mcts_samples = loader.get_mcts_training_samples()
        assert len(mcts_samples) == 1  # Only strategic_planning

    def test_get_dataset_summary(self, temp_cache_dir):
        """Test dataset summary generation."""
        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir)
        loader._all_samples = [
            DatasetSample(id="1", text="test", domain="abstract_reasoning", metadata={"source": "ARC"}),
            DatasetSample(id="2", text="test", domain="mathematics", metadata={"source": "GSM8K"}),
            DatasetSample(
                id="3", text="test", domain="strategic_planning", metadata={"source": "ChessGames"}
            ),
        ]

        summary = loader.get_dataset_summary()

        assert summary["total_samples"] == 3
        assert "abstract_reasoning" in summary["domains"]
        assert "mathematics" in summary["domains"]
        assert "strategic_planning" in summary["domains"]
        assert summary["hrm_training_samples"] == 2  # ARC + GSM8K
        assert summary["mcts_training_samples"] == 1  # Chess

    def test_export_for_training(self, temp_cache_dir):
        """Test exporting combined dataset."""
        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir)
        loader._all_samples = [
            DatasetSample(id="1", text="test1", domain="mathematics", metadata={"source": "GSM8K"}),
            DatasetSample(id="2", text="test2", domain="code_generation", metadata={"source": "HumanEval"}),
        ]

        output_path = Path(temp_cache_dir) / "export.jsonl"
        result_path = loader.export_for_training(str(output_path), format="jsonl")

        assert Path(result_path).exists()

        # Verify exported content
        with open(result_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            record1 = json.loads(lines[0])
            assert record1["id"] == "1"
            assert record1["domain"] == "mathematics"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDatasetIntegration:
    """Integration tests for dataset loaders."""

    @patch("src.data.dataset_loader.load_dataset")
    def test_end_to_end_training_pipeline(
        self, mock_load_dataset, temp_cache_dir, mock_gsm8k_dataset
    ):
        """Test end-to-end training data pipeline."""
        mock_load_dataset.return_value = mock_gsm8k_dataset

        # Load dataset
        loader = GSM8KLoader(cache_dir=temp_cache_dir)
        samples = loader.load(split="train")

        # Get statistics
        stats = loader.get_statistics()
        assert stats.total_samples == len(samples)

        # Iterate in batches
        batch_size = 1
        batches = list(loader.iterate_samples(batch_size=batch_size))
        assert len(batches) == len(samples)

        # Filter reasoning samples
        reasoning_samples = loader.get_reasoning_samples()
        assert all(len(s.reasoning_steps) > 1 for s in reasoning_samples)

    @patch("src.data.dataset_loader.load_dataset")
    def test_combined_loader_integration(
        self,
        mock_load_dataset,
        temp_cache_dir,
        mock_arc_dataset,
        mock_gsm8k_dataset,
        idoft_test_data,
    ):
        """Test combined loader with multiple datasets."""

        def mock_load_side_effect(dataset_name, *args, **kwargs):
            if "arc" in dataset_name.lower():
                return mock_arc_dataset
            elif "gsm8k" in dataset_name.lower():
                return mock_gsm8k_dataset
            return {"train": []}

        mock_load_dataset.side_effect = mock_load_side_effect

        # Load combined datasets
        loader = CombinedDatasetLoader(cache_dir=temp_cache_dir, idoft_data_path=idoft_test_data)

        # Load with multiple datasets enabled
        with patch("src.data.dataset_loader.DABStepLoader.load", return_value=[]):
            with patch("src.data.dataset_loader.PRIMUSLoader.load_seed", return_value=[]):
                samples = loader.load_all(
                    primus_max_samples=0,
                    include_instruct=False,
                    include_arc=True,
                    include_gsm8k=True,
                    include_idoft=True,
                )

        # Verify samples from different sources
        assert len(samples) > 0

        # Test domain distribution
        domain_dist = loader.get_domain_distribution()
        assert isinstance(domain_dist, dict)

        # Test agent-specific filtering
        hrm_samples = loader.get_hrm_training_samples()
        trm_samples = loader.get_trm_training_samples()
        assert isinstance(hrm_samples, list)
        assert isinstance(trm_samples, list)


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize(
    "loader_class,mock_dataset_fixture",
    [
        (ARCLoader, "mock_arc_dataset"),
        (GSM8KLoader, "mock_gsm8k_dataset"),
        (HumanEvalLoader, "mock_humaneval_dataset"),
        (BIGBenchHardLoader, "mock_bbh_dataset"),
    ],
)
@patch("src.data.dataset_loader.load_dataset")
def test_loader_basic_functionality(
    mock_load_dataset, loader_class, mock_dataset_fixture, temp_cache_dir, request
):
    """Parametrized test for basic loader functionality."""
    mock_dataset = request.getfixturevalue(mock_dataset_fixture)
    mock_load_dataset.return_value = mock_dataset

    loader = loader_class(cache_dir=temp_cache_dir)

    # Test load
    if loader_class == ARCLoader:
        samples = loader.load(split="train")
    elif loader_class == GSM8KLoader:
        samples = loader.load(split="train", config="main")
    elif loader_class == HumanEvalLoader:
        samples = loader.load(split="test")
    elif loader_class == BIGBenchHardLoader:
        samples = loader.load(split="train")

    assert len(samples) > 0
    assert all(isinstance(s, DatasetSample) for s in samples)

    # Test statistics
    stats = loader.get_statistics()
    assert isinstance(stats, DatasetStatistics)
    assert stats.total_samples == len(samples)

    # Test iteration
    batches = list(loader.iterate_samples(batch_size=1))
    assert len(batches) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in dataset loaders."""

    def test_statistics_before_load_raises_error(self, temp_cache_dir):
        """Test that getting statistics before loading raises error."""
        loader = ARCLoader(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="No samples loaded"):
            loader.get_statistics()

    def test_iteration_before_load_raises_error(self, temp_cache_dir):
        """Test that iteration before loading raises error."""
        loader = ARCLoader(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="No samples loaded"):
            list(loader.iterate_samples())

    @patch("src.data.dataset_loader.load_dataset")
    def test_dataset_import_error(self, mock_load_dataset, temp_cache_dir):
        """Test handling of missing datasets library."""
        mock_load_dataset.side_effect = ImportError("datasets library not installed")

        loader = ARCLoader(cache_dir=temp_cache_dir)

        with pytest.raises(ImportError):
            loader.load()


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests for dataset loaders (marked as slow)."""

    @patch("src.data.dataset_loader.load_dataset")
    def test_large_batch_iteration(self, mock_load_dataset, temp_cache_dir):
        """Test iteration over large batches."""
        # Create large mock dataset
        large_dataset = {
            "train": [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(1000)]
        }
        mock_load_dataset.return_value = large_dataset

        loader = GSM8KLoader(cache_dir=temp_cache_dir)
        samples = loader.load()

        # Test large batch sizes
        for batch_size in [10, 50, 100]:
            batches = list(loader.iterate_samples(batch_size=batch_size))
            total_samples = sum(len(batch) for batch in batches)
            assert total_samples == len(samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
