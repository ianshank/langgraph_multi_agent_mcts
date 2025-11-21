"""
Dataset Loading Module for Open-Source Training Data.

Provides unified loading interfaces for:
- DABStep: Multi-step data analysis reasoning
- PRIMUS: Cybersecurity domain knowledge
- ARC: Abstraction and Reasoning Corpus (for HRM training)
- GSM8K: Grade School Math (for mathematical reasoning)
- IDoFT: Illinois Dataset of Flaky Tests (for quality engineering)
- HumanEval: Code generation benchmark
- Chess Games: Strategic planning and MCTS training
- BIG-Bench Hard: Complex reasoning evaluation
- Custom tactical datasets

License Attribution:
- DABStep: CC-BY-4.0 (Creative Commons Attribution)
- PRIMUS: ODC-BY (Open Data Commons Attribution)
- ARC: Apache 2.0
- GSM8K: MIT License
- HumanEval: MIT License
- Chess Games: CC-BY-4.0
- BIG-Bench Hard: Apache 2.0
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Standardized representation of a dataset sample."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    labels: list[str] | None = None
    difficulty: str | None = None
    domain: str | None = None
    reasoning_steps: list[str] | None = None


@dataclass
class DatasetStatistics:
    """Statistics about a loaded dataset."""

    total_samples: int
    domains: dict[str, int]
    avg_text_length: float
    difficulty_distribution: dict[str, int]
    total_tokens: int = 0


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize dataset loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "mcts_datasets")
        self._dataset = None
        self._statistics = None

    @abstractmethod
    def load(self, split: str = "train") -> list[DatasetSample]:
        """Load dataset split."""
        pass

    @abstractmethod
    def get_statistics(self) -> DatasetStatistics:
        """Get dataset statistics."""
        pass

    @abstractmethod
    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        pass


class DABStepLoader(DatasetLoader):
    """
    Loader for DABStep Multi-Step Reasoning Dataset.

    DABStep contains 450+ data analysis tasks requiring sequential,
    iterative problem-solving. Perfect for training HRM/TRM agents.

    License: CC-BY-4.0 (Attribution required)
    Source: huggingface.co/datasets/adyen/DABstep
    """

    DATASET_NAME = "adyen/DABstep"
    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, cache_dir: str | None = None):
        """Initialize DABStep loader."""
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []

    def load(self, split: str = "train", difficulty: str | None = None) -> list[DatasetSample]:
        """
        Load DABStep dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            difficulty: Filter by difficulty ('easy', 'medium', 'hard')

        Returns:
            List of DatasetSample objects
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading DABStep dataset (split={split})")

            dataset = load_dataset(
                self.DATASET_NAME,
                cache_dir=self.cache_dir,
            )

            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available: {available_splits}")
                split = available_splits[0] if available_splits else "train"

            samples = []
            for idx, item in enumerate(dataset[split]):
                sample = DatasetSample(
                    id=f"dabstep_{split}_{idx}",
                    text=str(item.get("question", item.get("text", ""))),
                    metadata={
                        "source": "DABStep",
                        "license": "CC-BY-4.0",
                        "split": split,
                        "original_data": item,
                    },
                    difficulty=item.get("difficulty", "medium"),
                    domain="data_analysis",
                    reasoning_steps=item.get("steps", []),
                )

                if difficulty and sample.difficulty != difficulty:
                    continue

                samples.append(sample)

            self._loaded_samples = samples
            logger.info(f"Loaded {len(samples)} DABStep samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load DABStep: {e}")
            raise

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded DABStep data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        difficulty_dist = {}
        total_length = 0

        for sample in self._loaded_samples:
            diff = sample.difficulty or "unknown"
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
            total_length += len(sample.text)

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"data_analysis": len(self._loaded_samples)},
            avg_text_length=total_length / len(self._loaded_samples),
            difficulty_distribution=difficulty_dist,
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]

    def get_reasoning_tasks(self) -> list[DatasetSample]:
        """Get only samples with explicit reasoning steps."""
        return [s for s in self._loaded_samples if s.reasoning_steps]


class PRIMUSLoader(DatasetLoader):
    """
    Loader for PRIMUS Cybersecurity Dataset Suite.

    PRIMUS contains:
    - Seed: 674,848 cybersecurity documents (190M tokens)
    - Instruct: 835 instruction-tuning samples
    - Reasoning: Self-reflection data for reasoning

    License: ODC-BY (Open Data Commons Attribution)
    Source: huggingface.co/datasets/trendmicro-ailab/Primus-Seed
    """

    SEED_DATASET = "trendmicro-ailab/Primus-Seed"
    INSTRUCT_DATASET = "trendmicro-ailab/Primus-Instruct"

    DOMAINS = [
        "mitre_attack",
        "wikipedia",
        "company_sites",
        "threat_intelligence",
        "vulnerability_db",
    ]

    def __init__(self, cache_dir: str | None = None):
        """Initialize PRIMUS loader."""
        super().__init__(cache_dir)
        self._seed_samples: list[DatasetSample] = []
        self._instruct_samples: list[DatasetSample] = []

    def load(
        self,
        split: str = "train",
        dataset_type: str = "seed",
        domains: list[str] | None = None,
        max_samples: int | None = None,
        streaming: bool = True,
    ) -> list[DatasetSample]:
        """
        Load PRIMUS dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            dataset_type: 'seed' for knowledge base, 'instruct' for fine-tuning
            domains: Filter by specific domains
            max_samples: Limit number of samples (useful for large datasets)
            streaming: Use streaming mode for large datasets (default True)

        Returns:
            List of DatasetSample objects
        """
        try:
            from datasets import load_dataset

            dataset_name = self.SEED_DATASET if dataset_type == "seed" else self.INSTRUCT_DATASET

            logger.info(f"Loading PRIMUS {dataset_type} dataset")

            # Use streaming for large seed dataset to avoid download issues
            use_streaming = streaming and dataset_type == "seed" and max_samples is not None

            if use_streaming:
                logger.info(f"Using streaming mode (max_samples={max_samples})")
                dataset = load_dataset(
                    dataset_name,
                    "default",
                    streaming=True,
                    cache_dir=self.cache_dir,
                )
                # For streaming, iterate the first available split
                data_iter = iter(dataset["train"]) if "train" in dataset else iter(dataset[list(dataset.keys())[0]])
            else:
                dataset = load_dataset(
                    dataset_name,
                    cache_dir=self.cache_dir,
                )

                if split not in dataset:
                    available_splits = list(dataset.keys())
                    logger.warning(f"Split '{split}' not found. Using: {available_splits[0]}")
                    split = available_splits[0]

                data_iter = iter(dataset[split])

            samples = []
            count = 0

            for idx, item in enumerate(data_iter):
                if max_samples and count >= max_samples:
                    break

                domain = item.get("domain", item.get("source", "unknown"))

                if domains and domain not in domains:
                    continue

                if dataset_type == "instruct":
                    text = f"Instruction: {item.get('instruction', '')}\nResponse: {item.get('response', '')}"
                else:
                    text = str(item.get("text", item.get("content", "")))

                sample = DatasetSample(
                    id=f"primus_{dataset_type}_{split}_{idx}",
                    text=text,
                    metadata={
                        "source": f"PRIMUS-{dataset_type.capitalize()}",
                        "license": "ODC-BY",
                        "split": split,
                        "original_domain": domain,
                    },
                    domain=domain,
                    labels=item.get("labels", item.get("tags", [])),
                )

                samples.append(sample)
                count += 1

            if dataset_type == "seed":
                self._seed_samples = samples
            else:
                self._instruct_samples = samples

            logger.info(f"Loaded {len(samples)} PRIMUS {dataset_type} samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            if "gated dataset" in str(e):
                logger.error(
                    f"PRIMUS is a gated dataset. Please authenticate with HuggingFace:\n"
                    f"1. Create account at https://huggingface.co/\n"
                    f"2. Accept dataset terms at https://huggingface.co/datasets/{dataset_name}\n"
                    f"3. Create token at https://huggingface.co/settings/tokens\n"
                    f"4. Run: huggingface-cli login"
                )
            else:
                logger.error(f"Failed to load PRIMUS: {e}")
            raise

    def load_seed(self, max_samples: int | None = None) -> list[DatasetSample]:
        """Load PRIMUS-Seed knowledge base."""
        return self.load(dataset_type="seed", max_samples=max_samples)

    def load_instruct(self) -> list[DatasetSample]:
        """Load PRIMUS-Instruct fine-tuning data."""
        return self.load(dataset_type="instruct", streaming=False)

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded PRIMUS data."""
        all_samples = self._seed_samples + self._instruct_samples

        if not all_samples:
            raise ValueError("No samples loaded. Call load() first.")

        domain_dist = {}
        total_length = 0

        for sample in all_samples:
            domain = sample.domain or "unknown"
            domain_dist[domain] = domain_dist.get(domain, 0) + 1
            total_length += len(sample.text)

        return DatasetStatistics(
            total_samples=len(all_samples),
            domains=domain_dist,
            avg_text_length=total_length / len(all_samples),
            difficulty_distribution={"cybersecurity": len(all_samples)},
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over all loaded samples in batches."""
        all_samples = self._seed_samples + self._instruct_samples

        if not all_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(all_samples), batch_size):
            yield all_samples[i : i + batch_size]

    def get_mitre_attack_samples(self) -> list[DatasetSample]:
        """Get samples specifically from MITRE ATT&CK."""
        return [s for s in self._seed_samples if "mitre" in (s.domain or "").lower()]

    def get_threat_intelligence_samples(self) -> list[DatasetSample]:
        """Get threat intelligence related samples."""
        return [
            s
            for s in self._seed_samples
            if any(kw in (s.domain or "").lower() for kw in ["threat", "cti", "intelligence"])
        ]


class ARCLoader(DatasetLoader):
    """
    Loader for ARC (Abstraction and Reasoning Corpus) Dataset.

    ARC contains ~1,000 training examples and 400 evaluation tasks designed
    to measure abstract pattern recognition and hierarchical reasoning.
    Perfect for training HRM (Hierarchical Reasoning Model) agents.

    License: Apache 2.0
    Source: https://github.com/fchollet/ARC
    HuggingFace: Various implementations available
    """

    DATASET_NAME = "barc0/abstraction_and_reasoning_corpus"  # Community dataset

    def __init__(self, cache_dir: str | None = None):
        """Initialize ARC loader."""
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []

    def load(self, split: str = "train") -> list[DatasetSample]:
        """
        Load ARC dataset.

        Args:
            split: Dataset split ('train', 'evaluation', or 'test')

        Returns:
            List of DatasetSample objects with visual reasoning tasks
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading ARC dataset (split={split})")

            dataset = load_dataset(
                self.DATASET_NAME,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available: {available_splits}")
                split = available_splits[0] if available_splits else "train"

            samples = []
            for idx, item in enumerate(dataset[split]):
                # ARC tasks consist of input-output demonstration pairs
                task_data = {
                    "input_grids": item.get("train", {}).get("input", []),
                    "output_grids": item.get("train", {}).get("output", []),
                    "test_input": item.get("test", {}).get("input", []),
                    "test_output": item.get("test", {}).get("output", []),
                }

                # Create text description of the task
                text = f"ARC Visual Reasoning Task {idx}: Pattern recognition with {len(task_data['input_grids'])} training examples"

                sample = DatasetSample(
                    id=f"arc_{split}_{idx}",
                    text=text,
                    metadata={
                        "source": "ARC",
                        "license": "Apache-2.0",
                        "split": split,
                        "task_data": task_data,
                        "num_train_pairs": len(task_data["input_grids"]),
                    },
                    difficulty="hard",  # ARC is inherently challenging
                    domain="abstract_reasoning",
                    reasoning_steps=None,  # Visual pattern tasks
                )

                samples.append(sample)

            self._loaded_samples = samples
            logger.info(f"Loaded {len(samples)} ARC samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load ARC: {e}")
            raise

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded ARC data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"abstract_reasoning": len(self._loaded_samples)},
            avg_text_length=sum(len(s.text) for s in self._loaded_samples) / len(self._loaded_samples),
            difficulty_distribution={"hard": len(self._loaded_samples)},
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]


class GSM8KLoader(DatasetLoader):
    """
    Loader for GSM8K (Grade School Math 8K) Dataset.

    Contains 8,500 high-quality grade school math word problems requiring
    multi-step reasoning. Perfect for training TRM (iterative refinement)
    and testing consensus evaluation.

    Dataset: 7,500 training + 1,000 test problems

    License: MIT
    Source: https://github.com/openai/grade-school-math
    HuggingFace: openai/gsm8k
    """

    DATASET_NAME = "openai/gsm8k"

    def __init__(self, cache_dir: str | None = None):
        """Initialize GSM8K loader."""
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []

    def load(self, split: str = "train", config: str = "main") -> list[DatasetSample]:
        """
        Load GSM8K dataset.

        Args:
            split: Dataset split ('train' or 'test')
            config: Dataset configuration ('main' or 'socratic')

        Returns:
            List of DatasetSample objects with math problems
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading GSM8K dataset (split={split}, config={config})")

            dataset = load_dataset(
                self.DATASET_NAME,
                config,
                cache_dir=self.cache_dir,
            )

            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available: {available_splits}")
                split = available_splits[0] if available_splits else "train"

            samples = []
            for idx, item in enumerate(dataset[split]):
                question = item.get("question", "")
                answer = item.get("answer", "")

                # Extract reasoning steps from answer (GSM8K answers include step-by-step solutions)
                reasoning_steps = []
                if answer:
                    # Split by newlines or common delimiters
                    steps = [s.strip() for s in answer.split("\n") if s.strip()]
                    reasoning_steps = steps

                sample = DatasetSample(
                    id=f"gsm8k_{config}_{split}_{idx}",
                    text=question,
                    metadata={
                        "source": "GSM8K",
                        "license": "MIT",
                        "split": split,
                        "config": config,
                        "answer": answer,
                    },
                    difficulty="medium",  # Grade school level
                    domain="mathematics",
                    reasoning_steps=reasoning_steps,
                )

                samples.append(sample)

            self._loaded_samples = samples
            logger.info(f"Loaded {len(samples)} GSM8K samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            raise

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded GSM8K data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"mathematics": len(self._loaded_samples)},
            avg_text_length=sum(len(s.text) for s in self._loaded_samples) / len(self._loaded_samples),
            difficulty_distribution={"medium": len(self._loaded_samples)},
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]

    def get_reasoning_samples(self) -> list[DatasetSample]:
        """Get samples with multi-step reasoning."""
        return [s for s in self._loaded_samples if s.reasoning_steps and len(s.reasoning_steps) > 1]


class IDoFTLoader(DatasetLoader):
    """
    Loader for IDoFT (Illinois Dataset of Flaky Tests).

    Contains 2,000+ flaky tests from real-world projects with root cause
    classifications. Perfect for training quality engineering agents.

    Categories: async wait, concurrency, time-dependent, unordered collections,
    test order dependencies

    License: Varies by project (typically MIT/Apache 2.0)
    Source: http://mir.cs.illinois.edu/flakytests/
    GitHub: https://github.com/TestingResearchIllinois/idoft
    """

    DATASET_NAME = "flaky-tests/idoft"  # May need custom loading from GitHub

    FLAKY_CATEGORIES = [
        "async_wait",
        "concurrency",
        "time_dependent",
        "unordered_collection",
        "test_order_dependency",
        "resource_leak",
        "network_dependent",
        "io_dependent",
    ]

    def __init__(self, cache_dir: str | None = None, data_path: str | None = None):
        """
        Initialize IDoFT loader.

        Args:
            cache_dir: Directory to cache data
            data_path: Optional path to local IDoFT clone or data files
        """
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []
        self.data_path = data_path

    def load(self, split: str = "train", categories: list[str] | None = None) -> list[DatasetSample]:
        """
        Load IDoFT dataset.

        Note: IDoFT may require manual download from GitHub or research website.
        This loader attempts HuggingFace first, then falls back to local data.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            categories: Filter by flaky test categories

        Returns:
            List of DatasetSample objects with flaky test data
        """
        try:
            # Try loading from HuggingFace first
            try:
                from datasets import load_dataset

                logger.info(f"Attempting to load IDoFT from HuggingFace")
                dataset = load_dataset(self.DATASET_NAME, cache_dir=self.cache_dir)
                return self._process_hf_dataset(dataset, split, categories)
            except Exception as e:
                logger.warning(f"HuggingFace load failed: {e}. Trying local data...")

            # Fall back to local data if provided
            if self.data_path:
                return self._load_from_local(self.data_path, split, categories)
            else:
                logger.error(
                    "IDoFT dataset not available. Please either:\n"
                    "1. Clone from GitHub: git clone https://github.com/TestingResearchIllinois/idoft\n"
                    "2. Download from: http://mir.cs.illinois.edu/flakytests/\n"
                    "3. Provide data_path parameter with local data location"
                )
                raise FileNotFoundError("IDoFT dataset not found")

        except Exception as e:
            logger.error(f"Failed to load IDoFT: {e}")
            raise

    def _process_hf_dataset(
        self, dataset: Any, split: str, categories: list[str] | None
    ) -> list[DatasetSample]:
        """Process dataset from HuggingFace."""
        if split not in dataset:
            available_splits = list(dataset.keys())
            logger.warning(f"Split '{split}' not found. Available: {available_splits}")
            split = available_splits[0]

        samples = []
        for idx, item in enumerate(dataset[split]):
            category = item.get("category", item.get("root_cause", "unknown"))

            if categories and category not in categories:
                continue

            sample = DatasetSample(
                id=f"idoft_{split}_{idx}",
                text=item.get("test_code", item.get("description", "")),
                metadata={
                    "source": "IDoFT",
                    "license": "Research/Academic",
                    "split": split,
                    "project": item.get("project", ""),
                    "test_name": item.get("test_name", ""),
                    "failure_rate": item.get("failure_rate", 0.0),
                    "fix_commit": item.get("fix_commit", ""),
                },
                difficulty="hard",
                domain="quality_engineering",
                labels=[category],
            )

            samples.append(sample)

        self._loaded_samples = samples
        logger.info(f"Loaded {len(samples)} IDoFT samples")
        return samples

    def _load_from_local(self, data_path: str, split: str, categories: list[str] | None) -> list[DatasetSample]:
        """Load IDoFT from local files."""
        import json
        from pathlib import Path

        data_dir = Path(data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Look for JSON or CSV files
        json_files = list(data_dir.glob("**/*.json"))
        samples = []

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Handle both single dict and list of dicts
                    if isinstance(data, dict):
                        data = [data]

                    for idx, item in enumerate(data):
                        category = item.get("category", item.get("root_cause", "unknown"))

                        if categories and category not in categories:
                            continue

                        sample = DatasetSample(
                            id=f"idoft_local_{json_file.stem}_{idx}",
                            text=item.get("test_code", item.get("description", "")),
                            metadata={
                                "source": "IDoFT-Local",
                                "license": "Research/Academic",
                                "file": str(json_file),
                                "project": item.get("project", ""),
                                "test_name": item.get("test_name", ""),
                            },
                            difficulty="hard",
                            domain="quality_engineering",
                            labels=[category],
                        )

                        samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

        self._loaded_samples = samples
        logger.info(f"Loaded {len(samples)} IDoFT samples from local files")
        return samples

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded IDoFT data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        category_dist = {}
        for sample in self._loaded_samples:
            if sample.labels:
                for label in sample.labels:
                    category_dist[label] = category_dist.get(label, 0) + 1

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"quality_engineering": len(self._loaded_samples)},
            avg_text_length=sum(len(s.text) for s in self._loaded_samples) / len(self._loaded_samples),
            difficulty_distribution=category_dist,
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]

    def get_by_category(self, category: str) -> list[DatasetSample]:
        """Get samples by flaky test category."""
        return [s for s in self._loaded_samples if category in (s.labels or [])]


class HumanEvalLoader(DatasetLoader):
    """
    Loader for HumanEval Dataset.

    Contains 164 hand-crafted programming problems with function signatures,
    docstrings, and unit tests. Perfect for code generation evaluation and
    test generation training.

    License: MIT
    Source: https://github.com/openai/human-eval
    HuggingFace: openai/openai_humaneval
    """

    DATASET_NAME = "openai/openai_humaneval"

    def __init__(self, cache_dir: str | None = None):
        """Initialize HumanEval loader."""
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []

    def load(self, split: str = "test") -> list[DatasetSample]:
        """
        Load HumanEval dataset.

        Args:
            split: Dataset split (typically 'test' for HumanEval)

        Returns:
            List of DatasetSample objects with programming problems
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading HumanEval dataset")

            dataset = load_dataset(
                self.DATASET_NAME,
                cache_dir=self.cache_dir,
            )

            # HumanEval typically only has 'test' split
            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available: {available_splits}")
                split = available_splits[0]

            samples = []
            for idx, item in enumerate(dataset[split]):
                task_id = item.get("task_id", f"HumanEval/{idx}")
                prompt = item.get("prompt", "")
                canonical_solution = item.get("canonical_solution", "")
                test = item.get("test", "")
                entry_point = item.get("entry_point", "")

                # Combine prompt and tests as the "task"
                text = f"{prompt}\n# Tests:\n{test}"

                sample = DatasetSample(
                    id=f"humaneval_{task_id.replace('/', '_')}",
                    text=text,
                    metadata={
                        "source": "HumanEval",
                        "license": "MIT",
                        "task_id": task_id,
                        "prompt": prompt,
                        "canonical_solution": canonical_solution,
                        "test": test,
                        "entry_point": entry_point,
                    },
                    difficulty="medium",
                    domain="code_generation",
                    reasoning_steps=None,  # Code tasks
                )

                samples.append(sample)

            self._loaded_samples = samples
            logger.info(f"Loaded {len(samples)} HumanEval samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            raise

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded HumanEval data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"code_generation": len(self._loaded_samples)},
            avg_text_length=sum(len(s.text) for s in self._loaded_samples) / len(self._loaded_samples),
            difficulty_distribution={"medium": len(self._loaded_samples)},
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]


class ChessGamesLoader(DatasetLoader):
    """
    Loader for Chess Games Dataset.

    Contains 14 million chess games from high-level players (mean ELO 2388)
    with move sequences in UCI and SAN notation. Perfect for training MCTS
    policies and strategic planning.

    License: CC-BY-4.0
    Source: https://huggingface.co/datasets/angeluriot/chess_games
    """

    DATASET_NAME = "angeluriot/chess_games"

    def __init__(self, cache_dir: str | None = None):
        """Initialize Chess Games loader."""
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []

    def load(
        self,
        split: str = "train",
        min_elo: int = 2000,
        max_samples: int | None = None,
        streaming: bool = True,
    ) -> list[DatasetSample]:
        """
        Load Chess Games dataset.

        Args:
            split: Dataset split ('train', 'test', 'validation')
            min_elo: Minimum ELO rating to filter games
            max_samples: Maximum number of samples to load
            streaming: Use streaming mode for large dataset

        Returns:
            List of DatasetSample objects with chess games
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading Chess Games dataset (streaming={streaming})")

            if streaming and max_samples:
                dataset = load_dataset(
                    self.DATASET_NAME,
                    streaming=True,
                    cache_dir=self.cache_dir,
                )
                # Get the first available split
                data_iter = iter(dataset[split]) if split in dataset else iter(dataset[list(dataset.keys())[0]])
            else:
                dataset = load_dataset(
                    self.DATASET_NAME,
                    cache_dir=self.cache_dir,
                )

                if split not in dataset:
                    available_splits = list(dataset.keys())
                    logger.warning(f"Split '{split}' not found. Available: {available_splits}")
                    split = available_splits[0]

                data_iter = iter(dataset[split])

            samples = []
            count = 0

            for idx, item in enumerate(data_iter):
                if max_samples and count >= max_samples:
                    break

                white_elo = item.get("white_elo", 0)
                black_elo = item.get("black_elo", 0)

                # Filter by minimum ELO
                if white_elo < min_elo or black_elo < min_elo:
                    continue

                moves_uci = item.get("moves_uci", "")
                moves_san = item.get("moves_san", "")
                end_type = item.get("end_type", "unknown")

                # Create descriptive text
                text = f"Chess game (ELO: {white_elo} vs {black_elo}): {moves_san[:100]}..."

                sample = DatasetSample(
                    id=f"chess_{split}_{count}",
                    text=text,
                    metadata={
                        "source": "ChessGames",
                        "license": "CC-BY-4.0",
                        "split": split,
                        "date": item.get("date", ""),
                        "white_elo": white_elo,
                        "black_elo": black_elo,
                        "end_type": end_type,
                        "moves_uci": moves_uci,
                        "moves_san": moves_san,
                        "num_moves": len(moves_uci.split()) if moves_uci else 0,
                    },
                    difficulty="hard" if (white_elo + black_elo) / 2 > 2400 else "medium",
                    domain="strategic_planning",
                    reasoning_steps=moves_san.split() if moves_san else [],
                )

                samples.append(sample)
                count += 1

            self._loaded_samples = samples
            logger.info(f"Loaded {len(samples)} Chess game samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load Chess Games: {e}")
            raise

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded Chess data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        difficulty_dist = {}
        for sample in self._loaded_samples:
            diff = sample.difficulty or "unknown"
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"strategic_planning": len(self._loaded_samples)},
            avg_text_length=sum(len(s.text) for s in self._loaded_samples) / len(self._loaded_samples),
            difficulty_distribution=difficulty_dist,
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]

    def get_high_level_games(self, min_elo: int = 2500) -> list[DatasetSample]:
        """Get games from very high-level players."""
        return [
            s
            for s in self._loaded_samples
            if s.metadata.get("white_elo", 0) >= min_elo and s.metadata.get("black_elo", 0) >= min_elo
        ]


class BIGBenchHardLoader(DatasetLoader):
    """
    Loader for BIG-Bench Hard (BBH) Dataset.

    Contains 23 challenging BIG-Bench tasks requiring complex reasoning:
    causal reasoning, counterfactual thinking, multi-hop inference, etc.
    Perfect for evaluating HRM hierarchical reasoning capabilities.

    License: Apache 2.0
    Source: https://github.com/suzgunmirac/BIG-Bench-Hard
    HuggingFace: maveriq/bigbenchhard
    """

    DATASET_NAME = "maveriq/bigbenchhard"

    REASONING_TASKS = [
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction",
        "movie_recommendation",
        "multistep_arithmetic",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects",
        "web_of_lies",
    ]

    def __init__(self, cache_dir: str | None = None):
        """Initialize BIG-Bench Hard loader."""
        super().__init__(cache_dir)
        self._loaded_samples: list[DatasetSample] = []

    def load(self, split: str = "train", tasks: list[str] | None = None) -> list[DatasetSample]:
        """
        Load BIG-Bench Hard dataset.

        Args:
            split: Dataset split ('train' or 'validation')
            tasks: Filter by specific task names (None for all)

        Returns:
            List of DatasetSample objects with reasoning tasks
        """
        try:
            from datasets import load_dataset

            logger.info(f"Loading BIG-Bench Hard dataset (split={split})")

            dataset = load_dataset(
                self.DATASET_NAME,
                cache_dir=self.cache_dir,
            )

            if split not in dataset:
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{split}' not found. Available: {available_splits}")
                split = available_splits[0]

            samples = []
            for idx, item in enumerate(dataset[split]):
                task_name = item.get("task", item.get("task_name", "unknown"))

                if tasks and task_name not in tasks:
                    continue

                question = item.get("input", item.get("question", ""))
                target = item.get("target", item.get("answer", ""))

                sample = DatasetSample(
                    id=f"bbh_{task_name}_{split}_{idx}",
                    text=question,
                    metadata={
                        "source": "BIG-Bench-Hard",
                        "license": "Apache-2.0",
                        "split": split,
                        "task": task_name,
                        "target": target,
                    },
                    difficulty="hard",  # BBH is intentionally challenging
                    domain="complex_reasoning",
                    labels=[task_name],
                )

                samples.append(sample)

            self._loaded_samples = samples
            logger.info(f"Loaded {len(samples)} BIG-Bench Hard samples")
            return samples

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load BIG-Bench Hard: {e}")
            raise

    def get_statistics(self) -> DatasetStatistics:
        """Get statistics about loaded BBH data."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        task_dist = {}
        for sample in self._loaded_samples:
            if sample.labels:
                for label in sample.labels:
                    task_dist[label] = task_dist.get(label, 0) + 1

        return DatasetStatistics(
            total_samples=len(self._loaded_samples),
            domains={"complex_reasoning": len(self._loaded_samples)},
            avg_text_length=sum(len(s.text) for s in self._loaded_samples) / len(self._loaded_samples),
            difficulty_distribution=task_dist,
        )

    def iterate_samples(self, batch_size: int = 32) -> Iterator[list[DatasetSample]]:
        """Iterate over samples in batches."""
        if not self._loaded_samples:
            raise ValueError("No samples loaded. Call load() first.")

        for i in range(0, len(self._loaded_samples), batch_size):
            yield self._loaded_samples[i : i + batch_size]

    def get_by_task(self, task_name: str) -> list[DatasetSample]:
        """Get samples for a specific reasoning task."""
        return [s for s in self._loaded_samples if task_name in (s.labels or [])]


class CombinedDatasetLoader:
    """
    Unified loader for combining multiple datasets.

    Provides a single interface for loading and managing:
    - DABStep (multi-step reasoning)
    - PRIMUS (cybersecurity knowledge)
    - ARC (abstract reasoning for HRM training)
    - GSM8K (mathematical reasoning)
    - IDoFT (flaky tests for quality engineering)
    - HumanEval (code generation)
    - Chess Games (MCTS strategic planning)
    - BIG-Bench Hard (complex reasoning evaluation)
    - Custom tactical datasets
    """

    def __init__(self, cache_dir: str | None = None, idoft_data_path: str | None = None):
        """
        Initialize combined loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
            idoft_data_path: Optional path to local IDoFT data files
        """
        self.cache_dir = cache_dir
        self.dabstep_loader = DABStepLoader(cache_dir)
        self.primus_loader = PRIMUSLoader(cache_dir)
        self.arc_loader = ARCLoader(cache_dir)
        self.gsm8k_loader = GSM8KLoader(cache_dir)
        self.idoft_loader = IDoFTLoader(cache_dir, idoft_data_path)
        self.humaneval_loader = HumanEvalLoader(cache_dir)
        self.chess_loader = ChessGamesLoader(cache_dir)
        self.bbh_loader = BIGBenchHardLoader(cache_dir)
        self._all_samples: list[DatasetSample] = []

    def load_all(
        self,
        # Original datasets
        dabstep_split: str = "train",
        primus_max_samples: int | None = 10000,
        include_instruct: bool = True,
        # New datasets
        include_arc: bool = False,
        arc_split: str = "train",
        include_gsm8k: bool = False,
        gsm8k_split: str = "train",
        include_idoft: bool = False,
        include_humaneval: bool = False,
        include_chess: bool = False,
        chess_max_samples: int | None = 1000,
        chess_min_elo: int = 2000,
        include_bbh: bool = False,
        bbh_split: str = "train",
    ) -> list[DatasetSample]:
        """
        Load all requested datasets.

        Args:
            dabstep_split: Split for DABStep
            primus_max_samples: Max samples from PRIMUS-Seed (None for all)
            include_instruct: Whether to include PRIMUS-Instruct
            include_arc: Include ARC dataset for HRM training
            arc_split: Split for ARC dataset
            include_gsm8k: Include GSM8K for mathematical reasoning
            gsm8k_split: Split for GSM8K
            include_idoft: Include IDoFT for flaky test analysis
            include_humaneval: Include HumanEval for code generation
            include_chess: Include Chess games for MCTS training
            chess_max_samples: Max chess games to load
            chess_min_elo: Minimum ELO rating for chess games
            include_bbh: Include BIG-Bench Hard for reasoning evaluation
            bbh_split: Split for BIG-Bench Hard

        Returns:
            Combined list of all samples
        """
        logger.info("Loading combined datasets")
        all_samples = []

        # Load DABStep
        dabstep_samples = self.dabstep_loader.load(split=dabstep_split)
        logger.info(f"DABStep: {len(dabstep_samples)} samples")
        all_samples.extend(dabstep_samples)

        # Load PRIMUS-Seed
        primus_seed = self.primus_loader.load_seed(max_samples=primus_max_samples)
        logger.info(f"PRIMUS-Seed: {len(primus_seed)} samples")
        all_samples.extend(primus_seed)

        # Load PRIMUS-Instruct
        if include_instruct:
            primus_instruct = self.primus_loader.load_instruct()
            logger.info(f"PRIMUS-Instruct: {len(primus_instruct)} samples")
            all_samples.extend(primus_instruct)

        # Load ARC
        if include_arc:
            try:
                arc_samples = self.arc_loader.load(split=arc_split)
                logger.info(f"ARC: {len(arc_samples)} samples")
                all_samples.extend(arc_samples)
            except Exception as e:
                logger.warning(f"Failed to load ARC: {e}")

        # Load GSM8K
        if include_gsm8k:
            try:
                gsm8k_samples = self.gsm8k_loader.load(split=gsm8k_split)
                logger.info(f"GSM8K: {len(gsm8k_samples)} samples")
                all_samples.extend(gsm8k_samples)
            except Exception as e:
                logger.warning(f"Failed to load GSM8K: {e}")

        # Load IDoFT
        if include_idoft:
            try:
                idoft_samples = self.idoft_loader.load()
                logger.info(f"IDoFT: {len(idoft_samples)} samples")
                all_samples.extend(idoft_samples)
            except Exception as e:
                logger.warning(f"Failed to load IDoFT: {e}")

        # Load HumanEval
        if include_humaneval:
            try:
                humaneval_samples = self.humaneval_loader.load()
                logger.info(f"HumanEval: {len(humaneval_samples)} samples")
                all_samples.extend(humaneval_samples)
            except Exception as e:
                logger.warning(f"Failed to load HumanEval: {e}")

        # Load Chess Games
        if include_chess:
            try:
                chess_samples = self.chess_loader.load(
                    max_samples=chess_max_samples, min_elo=chess_min_elo, streaming=True
                )
                logger.info(f"Chess Games: {len(chess_samples)} samples")
                all_samples.extend(chess_samples)
            except Exception as e:
                logger.warning(f"Failed to load Chess Games: {e}")

        # Load BIG-Bench Hard
        if include_bbh:
            try:
                bbh_samples = self.bbh_loader.load(split=bbh_split)
                logger.info(f"BIG-Bench Hard: {len(bbh_samples)} samples")
                all_samples.extend(bbh_samples)
            except Exception as e:
                logger.warning(f"Failed to load BIG-Bench Hard: {e}")

        self._all_samples = all_samples
        logger.info(f"Total combined samples: {len(self._all_samples)}")

        return self._all_samples

    def get_domain_distribution(self) -> dict[str, int]:
        """Get distribution of samples across domains."""
        dist = {}
        for sample in self._all_samples:
            domain = sample.domain or "unknown"
            dist[domain] = dist.get(domain, 0) + 1
        return dist

    def filter_by_domain(self, domain: str) -> list[DatasetSample]:
        """Filter samples by domain."""
        return [s for s in self._all_samples if s.domain == domain]

    def get_multi_step_reasoning_samples(self) -> list[DatasetSample]:
        """Get samples suitable for multi-step reasoning training."""
        return [
            s
            for s in self._all_samples
            if s.reasoning_steps or s.domain == "data_analysis" or "instruct" in s.metadata.get("source", "").lower()
        ]

    def get_hrm_training_samples(self) -> list[DatasetSample]:
        """
        Get samples suitable for HRM (Hierarchical Reasoning Model) training.

        Includes: ARC, DABStep, GSM8K, BIG-Bench Hard
        """
        hrm_domains = {"abstract_reasoning", "data_analysis", "mathematics", "complex_reasoning"}
        return [s for s in self._all_samples if s.domain in hrm_domains]

    def get_trm_training_samples(self) -> list[DatasetSample]:
        """
        Get samples suitable for TRM (Task Refinement Model) training.

        Includes: GSM8K, PRIMUS-Instruct, DABStep, Code generation tasks
        """
        trm_sources = {"GSM8K", "PRIMUS-Instruct", "DABStep", "HumanEval"}
        return [s for s in self._all_samples if s.metadata.get("source") in trm_sources]

    def get_mcts_training_samples(self) -> list[DatasetSample]:
        """
        Get samples suitable for MCTS (Monte Carlo Tree Search) training.

        Includes: Chess games, strategic planning tasks
        """
        return [s for s in self._all_samples if s.domain == "strategic_planning"]

    def get_code_generation_samples(self) -> list[DatasetSample]:
        """
        Get samples for code generation and quality engineering training.

        Includes: HumanEval, IDoFT
        """
        code_domains = {"code_generation", "quality_engineering"}
        return [s for s in self._all_samples if s.domain in code_domains]

    def get_quality_engineering_samples(self) -> list[DatasetSample]:
        """Get samples specifically for quality engineering (flaky tests)."""
        return [s for s in self._all_samples if s.domain == "quality_engineering"]

    def get_mathematical_reasoning_samples(self) -> list[DatasetSample]:
        """Get samples for mathematical reasoning."""
        return [s for s in self._all_samples if s.domain == "mathematics"]

    def get_dataset_summary(self) -> dict[str, Any]:
        """
        Get comprehensive summary of loaded datasets.

        Returns:
            Dictionary with dataset statistics and breakdowns
        """
        domain_dist = self.get_domain_distribution()
        source_dist = {}
        for sample in self._all_samples:
            source = sample.metadata.get("source", "unknown")
            source_dist[source] = source_dist.get(source, 0) + 1

        return {
            "total_samples": len(self._all_samples),
            "domains": domain_dist,
            "sources": source_dist,
            "hrm_training_samples": len(self.get_hrm_training_samples()),
            "trm_training_samples": len(self.get_trm_training_samples()),
            "mcts_training_samples": len(self.get_mcts_training_samples()),
            "code_generation_samples": len(self.get_code_generation_samples()),
            "quality_engineering_samples": len(self.get_quality_engineering_samples()),
        }

    def export_for_training(self, output_path: str, format: str = "jsonl") -> str:
        """
        Export dataset for training.

        Args:
            output_path: Path to save exported data
            format: Export format ('jsonl', 'csv', 'parquet')

        Returns:
            Path to exported file
        """
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in self._all_samples:
                    record = {
                        "id": sample.id,
                        "text": sample.text,
                        "domain": sample.domain,
                        "difficulty": sample.difficulty,
                        "labels": sample.labels,
                        "metadata": sample.metadata,
                    }
                    f.write(json.dumps(record) + "\n")
        else:
            raise NotImplementedError(f"Format {format} not yet supported")

        logger.info(f"Exported {len(self._all_samples)} samples to {output_file}")
        return str(output_file)


def load_dataset(
    dataset_name: str,
    split: str = "train",
    cache_dir: str | None = None,
    **kwargs,
) -> Any:
    """
    Unified interface for loading datasets from HuggingFace.

    This function provides compatibility with the standard HuggingFace datasets API.
    It wraps the underlying load_dataset function from the datasets library.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "adyen/DABstep")
        split: Dataset split to load ("train", "validation", "test")
        cache_dir: Optional directory for caching downloaded datasets
        **kwargs: Additional arguments passed to datasets.load_dataset

    Returns:
        HuggingFace Dataset object or dict of Dataset objects

    Raises:
        ImportError: If datasets library is not installed
        Exception: If dataset loading fails

    Examples:
        >>> # Load DABStep dataset
        >>> dataset = load_dataset("adyen/DABstep")
        >>> samples = dataset["train"]

        >>> # Load PRIMUS-Seed with custom cache
        >>> dataset = load_dataset("trendmicro-ailab/Primus-Seed", cache_dir="/tmp/cache")

    License Attribution:
        - DABStep: CC-BY-4.0 (Creative Commons Attribution 4.0)
        - PRIMUS: ODC-BY (Open Data Commons Attribution)
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info(f"Loading dataset: {dataset_name} (split={split})")

        load_kwargs = {
            **kwargs,
        }

        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        dataset = hf_load_dataset(dataset_name, **load_kwargs)

        logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset

    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        raise ImportError("The datasets library is required but not installed. Install it with: pip install datasets")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
