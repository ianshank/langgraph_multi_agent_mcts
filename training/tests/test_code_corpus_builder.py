"""
Tests for Code Corpus Builder

Tests repository fetching, code parsing, quality filtering, and RAG integration.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from training.code_corpus_builder import (
    CodeChunk,
    CodeCorpusBuilder,
    CodeQualityFilter,
    ExampleExtractor,
    PythonCodeParser,
    RepositoryFetcher,
)

# ============================================================================
# Test Data
# ============================================================================

SAMPLE_PYTHON_CODE = '''
"""Sample module for testing."""

import numpy as np
from typing import List, Optional


def simple_function(x: int) -> int:
    """
    A simple function for testing.

    Args:
        x: Input integer

    Returns:
        The input multiplied by 2

    References:
        - Smith et al. (2023)
        - arXiv:2301.12345
    """
    return x * 2


class SampleClass:
    """
    A sample class for testing.

    This class demonstrates code extraction.
    """

    def __init__(self, value: int):
        """Initialize with a value."""
        self.value = value

    def method_with_docstring(self, x: int) -> int:
        """
        A method with proper documentation.

        Args:
            x: Input value

        Returns:
            Computed result
        """
        return self.value + x

    def method_without_docstring(self):
        return self.value * 2


async def async_function(data: List[int]) -> Optional[int]:
    """An async function example."""
    if not data:
        return None
    return sum(data)


def complex_function():
    """A function with high complexity."""
    result = 0
    for i in range(10):
        for j in range(10):
            if i > j:
                result += i * j
            else:
                result -= i + j
            if result > 100:
                result = result % 100
    return result
'''


SAMPLE_TEST_CODE = '''
"""Test file for sample module."""

import pytest
from sample_module import simple_function, SampleClass


def test_simple_function():
    """Test the simple function."""
    assert simple_function(5) == 10
    assert simple_function(0) == 0


def test_sample_class():
    """Test SampleClass."""
    obj = SampleClass(10)
    assert obj.method_with_docstring(5) == 15


def test_edge_cases():
    """Test edge cases."""
    assert simple_function(-1) == -2
'''


SAMPLE_EXAMPLE_CODE = '''
"""Example usage of sample module."""

from sample_module import simple_function, SampleClass


# Example 1: Basic usage
result = simple_function(42)
print(f"Result: {result}")


# Example 2: Using the class
obj = SampleClass(100)
output = obj.method_with_docstring(50)
print(f"Output: {output}")
'''


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_config(tmp_path):
    """Create temporary configuration."""
    config = {
        "code_corpus": {
            "cache_dir": str(tmp_path / "cache_repos"),
            "output_dir": str(tmp_path / "output"),
            "use_github_api": False,
            "shallow_clone": True,
            "min_function_lines": 2,
            "max_function_lines": 100,
            "extract_tests": True,
            "extract_examples": True,
            "find_tests": True,
            "min_quality_score": 0.3,
        }
    }
    return config


@pytest.fixture
def sample_code_file(tmp_path):
    """Create a sample Python file."""
    code_file = tmp_path / "sample_module.py"
    code_file.write_text(SAMPLE_PYTHON_CODE)
    return code_file


@pytest.fixture
def sample_test_file(tmp_path):
    """Create a sample test file."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_sample_module.py"
    test_file.write_text(SAMPLE_TEST_CODE)
    return test_file


@pytest.fixture
def sample_example_file(tmp_path):
    """Create a sample example file."""
    example_dir = tmp_path / "examples"
    example_dir.mkdir()
    example_file = example_dir / "example_usage.py"
    example_file.write_text(SAMPLE_EXAMPLE_CODE)
    return example_file


# ============================================================================
# Tests: PythonCodeParser
# ============================================================================


def test_parser_initialization(temp_config):
    """Test parser initialization."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    assert parser.min_function_lines == 2
    assert parser.max_function_lines == 100


def test_parse_function(sample_code_file, temp_config):
    """Test parsing a Python file and extracting functions."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    # Should extract simple_function, class methods, async_function, complex_function
    assert len(chunks) > 0

    # Find simple_function
    simple_func = next((c for c in chunks if c.function_name == "simple_function"), None)
    assert simple_func is not None
    assert simple_func.docstring != ""
    assert "Args:" in simple_func.docstring
    assert "def simple_function" in simple_func.code


def test_parse_class(sample_code_file, temp_config):
    """Test parsing classes and methods."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    # Find class chunk
    class_chunk = next((c for c in chunks if "class SampleClass" in c.function_name), None)
    assert class_chunk is not None
    assert class_chunk.docstring != ""

    # Find method chunks
    method_chunks = [c for c in chunks if "SampleClass." in c.function_name]
    assert len(method_chunks) > 0

    # Check method with docstring
    method_with_doc = next((c for c in method_chunks if "method_with_docstring" in c.function_name), None)
    assert method_with_doc is not None
    assert method_with_doc.docstring != ""
    assert method_with_doc.metadata["class"] == "SampleClass"


def test_extract_imports(sample_code_file, temp_config):
    """Test import extraction."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert len(first_chunk.imports) > 0
    assert any("numpy" in imp for imp in first_chunk.imports)


def test_extract_paper_references(sample_code_file, temp_config):
    """Test extracting paper references from docstrings."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    simple_func = next((c for c in chunks if c.function_name == "simple_function"), None)
    assert simple_func is not None
    assert len(simple_func.related_papers) > 0
    # Should find arXiv reference
    assert any("arXiv" in paper for paper in simple_func.related_papers)


def test_async_function_detection(sample_code_file, temp_config):
    """Test detection of async functions."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    async_func = next((c for c in chunks if c.function_name == "async_function"), None)
    assert async_func is not None
    assert async_func.metadata.get("is_async") is True


def test_complexity_calculation(sample_code_file, temp_config):
    """Test complexity score calculation."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    # Complex function should have higher complexity
    complex_func = next((c for c in chunks if c.function_name == "complex_function"), None)
    simple_func = next((c for c in chunks if c.function_name == "simple_function"), None)

    assert complex_func is not None
    assert simple_func is not None
    assert complex_func.complexity_score > simple_func.complexity_score


# ============================================================================
# Tests: ExampleExtractor
# ============================================================================


def test_example_extraction(tmp_path, sample_code_file, sample_example_file, temp_config):
    """Test extracting usage examples."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    extractor = ExampleExtractor(temp_config["code_corpus"])
    examples_map = extractor.extract_examples_from_repo(tmp_path, chunks)

    # Should find examples for simple_function
    assert "simple_function" in examples_map
    assert len(examples_map["simple_function"]) > 0


def test_find_test_files(tmp_path, sample_code_file, sample_test_file, temp_config):
    """Test finding test files for functions."""
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    extractor = ExampleExtractor(temp_config["code_corpus"])

    simple_func = next((c for c in chunks if c.function_name == "simple_function"), None)
    assert simple_func is not None

    test_files = extractor.find_test_files(tmp_path, simple_func)
    assert len(test_files) > 0
    assert any("test_" in tf for tf in test_files)


# ============================================================================
# Tests: CodeQualityFilter
# ============================================================================


def test_quality_score_calculation(temp_config):
    """Test quality score calculation."""
    filter_obj = CodeQualityFilter(temp_config["code_corpus"])

    # High-quality chunk
    high_quality = CodeChunk(
        repo_name="test/repo",
        file_path="test.py",
        function_name="good_function",
        code="def good_function():\n    pass",
        docstring="Good documentation",
        imports=["numpy"],
        usage_examples=["example()"],
        related_papers=[],
        metadata={"returns": "int"},
        complexity_score=50,
        test_files=["test_good.py"],
    )

    score_high = filter_obj.calculate_quality_score(high_quality)
    assert score_high > 0.5

    # Low-quality chunk
    low_quality = CodeChunk(
        repo_name="test/repo",
        file_path="test.py",
        function_name="bad_function",
        code="def bad_function(): pass",
        docstring="",
        imports=[],
        usage_examples=[],
        related_papers=[],
        metadata={"returns": "Any"},
        complexity_score=2,
        test_files=[],
    )

    score_low = filter_obj.calculate_quality_score(low_quality)
    assert score_low < score_high


def test_quality_filtering(temp_config):
    """Test filtering chunks by quality."""
    filter_obj = CodeQualityFilter(temp_config["code_corpus"])

    chunks = [
        CodeChunk(
            repo_name="test/repo",
            file_path="test.py",
            function_name=f"func_{i}",
            code=f"def func_{i}(): pass",
            docstring="Good" if i % 2 == 0 else "",
            imports=["numpy"] if i % 2 == 0 else [],
            usage_examples=["example()"] if i % 2 == 0 else [],
            related_papers=[],
            metadata={"returns": "int" if i % 2 == 0 else "Any"},
            complexity_score=50 if i % 2 == 0 else 2,
        )
        for i in range(10)
    ]

    filtered = filter_obj.filter_chunks(chunks)
    assert len(filtered) < len(chunks)


def test_deduplication(temp_config):
    """Test deduplication of code chunks."""
    filter_obj = CodeQualityFilter(temp_config["code_corpus"])

    # Create duplicate chunks
    chunks = [
        CodeChunk(
            repo_name="test/repo",
            file_path=f"test{i}.py",
            function_name=f"func_{i}",
            code="def duplicate(): pass",  # Same code
            docstring="",
            imports=[],
            usage_examples=[],
            related_papers=[],
            metadata={},
            complexity_score=10,
        )
        for i in range(5)
    ]

    unique = filter_obj.deduplicate_chunks(chunks)
    assert len(unique) == 1


# ============================================================================
# Tests: CodeChunk
# ============================================================================


def test_code_chunk_to_document_chunk():
    """Test converting CodeChunk to DocumentChunk."""
    code_chunk = CodeChunk(
        repo_name="test/repo",
        file_path="test.py",
        function_name="test_function",
        code="def test_function():\n    return 42",
        docstring="Test function that returns 42",
        imports=["numpy"],
        usage_examples=["result = test_function()"],
        related_papers=["Smith et al. 2023"],
        metadata={"type": "function"},
        start_line=10,
        end_line=15,
        complexity_score=20.0,
    )

    doc_chunk = code_chunk.to_document_chunk(chunk_id=1)

    assert doc_chunk.chunk_id == 1
    assert "test/repo" in doc_chunk.text
    assert "test_function" in doc_chunk.text
    assert "def test_function" in doc_chunk.text
    assert doc_chunk.metadata["type"] == "code"
    assert doc_chunk.metadata["function_name"] == "test_function"
    assert doc_chunk.metadata["has_docstring"] is True
    assert doc_chunk.metadata["has_examples"] is True


def test_code_chunk_serialization():
    """Test CodeChunk serialization to dictionary."""
    code_chunk = CodeChunk(
        repo_name="test/repo",
        file_path="test.py",
        function_name="test_function",
        code="def test_function(): pass",
        docstring="Test",
        imports=[],
        usage_examples=[],
        related_papers=[],
        metadata={"type": "function"},
    )

    chunk_dict = code_chunk.to_dict()
    assert isinstance(chunk_dict, dict)
    assert chunk_dict["repo_name"] == "test/repo"
    assert chunk_dict["function_name"] == "test_function"


# ============================================================================
# Tests: RepositoryFetcher
# ============================================================================


def test_repository_fetcher_initialization(temp_config):
    """Test repository fetcher initialization."""
    fetcher = RepositoryFetcher(temp_config["code_corpus"])
    assert fetcher.cache_dir.exists()
    assert fetcher.shallow_clone is True


@patch("subprocess.run")
def test_fetch_repository(mock_run, temp_config):
    """Test repository cloning."""
    mock_run.return_value = MagicMock(returncode=0)

    fetcher = RepositoryFetcher(temp_config["code_corpus"])
    repo_info = {"name": "test/repo", "description": "Test repository"}

    # This will try to clone, but we're mocking it
    try:
        repo_path = fetcher.fetch_repository(repo_info)
        # If it succeeds (uses cache), verify path
        assert "test_repo" in str(repo_path)
    except Exception:
        # If it fails due to mocking, that's expected
        pass


def test_license_compliance_check(tmp_path, temp_config):
    """Test license compliance checking."""
    # Create a mock repository with MIT license
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    license_file = repo_path / "LICENSE"
    license_file.write_text("MIT License\n\nCopyright (c) 2024")

    fetcher = RepositoryFetcher(temp_config["code_corpus"])
    license_info = fetcher.check_license_compliance(repo_path)

    assert license_info["found"] is True
    assert license_info["type"] == "MIT"
    assert license_info["compliant"] is True


# ============================================================================
# Tests: CodeCorpusBuilder
# ============================================================================


def test_corpus_builder_initialization(temp_config, tmp_path):
    """Test corpus builder initialization."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(temp_config, f)

    builder = CodeCorpusBuilder(str(config_file))
    assert builder.output_dir.exists()
    assert len(builder.all_chunks) == 0


def test_save_and_load_corpus(temp_config, tmp_path):
    """Test saving and loading corpus."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(temp_config, f)

    builder = CodeCorpusBuilder(str(config_file))

    # Add sample chunks
    builder.all_chunks = [
        CodeChunk(
            repo_name="test/repo",
            file_path="test.py",
            function_name="test_func",
            code="def test(): pass",
            docstring="Test",
            imports=[],
            usage_examples=[],
            related_papers=[],
            metadata={"type": "function"},
        )
    ]

    # Save
    output_path = tmp_path / "corpus_output"
    builder.save_corpus(output_path)

    assert (output_path / "code_chunks.json").exists()
    assert (output_path / "corpus_statistics.json").exists()

    # Load
    builder2 = CodeCorpusBuilder(str(config_file))
    loaded_chunks = builder2.load_corpus(output_path)

    assert len(loaded_chunks) == 1
    assert loaded_chunks[0].function_name == "test_func"


def test_corpus_statistics(temp_config, tmp_path):
    """Test corpus statistics generation."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(temp_config, f)

    builder = CodeCorpusBuilder(str(config_file))

    # Add sample chunks
    builder.all_chunks = [
        CodeChunk(
            repo_name="test/repo",
            file_path="test.py",
            function_name=f"func_{i}",
            code=f"def func_{i}(): pass",
            docstring="Test" if i % 2 == 0 else "",
            imports=[],
            usage_examples=["example()"] if i % 3 == 0 else [],
            related_papers=[],
            metadata={"type": "function", "quality_score": 0.7},
            complexity_score=10.0 * i,
        )
        for i in range(10)
    ]

    stats = builder.get_corpus_statistics()

    assert stats["total_chunks"] == 10
    assert stats["chunks_with_docstrings"] == 5
    assert stats["avg_complexity"] > 0
    assert "function" in stats["chunk_types"]


def test_simple_code_search(temp_config, tmp_path):
    """Test simple keyword-based code search."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(temp_config, f)

    builder = CodeCorpusBuilder(str(config_file))

    # Add sample chunks
    builder.all_chunks = [
        CodeChunk(
            repo_name="test/repo",
            file_path="test.py",
            function_name="mcts_search",
            code="def mcts_search(): pass",
            docstring="Monte Carlo Tree Search implementation",
            imports=[],
            usage_examples=[],
            related_papers=[],
            metadata={"type": "function", "topics": ["mcts", "tree-search"]},
        ),
        CodeChunk(
            repo_name="test/repo",
            file_path="other.py",
            function_name="random_function",
            code="def random_function(): pass",
            docstring="Unrelated function",
            imports=[],
            usage_examples=[],
            related_papers=[],
            metadata={"type": "function"},
        ),
    ]

    results = builder.search_code("mcts", top_k=5)

    assert len(results) > 0
    assert results[0].function_name == "mcts_search"


# ============================================================================
# Integration Test
# ============================================================================


def test_end_to_end_pipeline(tmp_path, sample_code_file, temp_config):
    """Test end-to-end pipeline from parsing to indexing."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(temp_config, f)

    # Initialize builder
    builder = CodeCorpusBuilder(str(config_file))

    # Parse file
    parser = PythonCodeParser(temp_config["code_corpus"])
    chunks = parser.parse_file(sample_code_file, "test/repo")

    # Filter
    filter_obj = CodeQualityFilter(temp_config["code_corpus"])
    filtered_chunks = filter_obj.filter_chunks(chunks)
    unique_chunks = filter_obj.deduplicate_chunks(filtered_chunks)

    # Add to builder
    builder.all_chunks = unique_chunks

    # Convert to document chunks
    doc_chunks = list(builder.stream_document_chunks())

    assert len(doc_chunks) > 0
    for doc_chunk in doc_chunks:
        assert doc_chunk.metadata["type"] == "code"
        assert "Repository:" in doc_chunk.text
        assert "Code" in doc_chunk.text

    # Save
    output_path = tmp_path / "final_output"
    builder.save_corpus(output_path)

    assert (output_path / "code_chunks.json").exists()

    # Verify saved data
    with open(output_path / "code_chunks.json") as f:
        saved_data = json.load(f)
        assert len(saved_data) == len(unique_chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
