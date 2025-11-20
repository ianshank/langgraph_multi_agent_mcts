"""
Example Usage of Benchmark Suite

This module provides practical examples of how to use the benchmark suite
for evaluating RAG systems, reasoning capabilities, and code generation.

Examples include:
1. Basic retrieval benchmark
2. Reasoning quality evaluation
3. Code generation assessment
4. A/B testing between models
5. CI/CD integration
"""

import logging
from pathlib import Path

from training.benchmark_suite import (
    BenchmarkSuite,
    RetrievalResult,
    create_example_reasoning_function,
    create_example_retrieval_function,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Basic Retrieval Benchmark
# =============================================================================


def example_basic_retrieval_benchmark():
    """Example: Run a basic retrieval benchmark."""
    print("=" * 70)
    print("Example 1: Basic Retrieval Benchmark")
    print("=" * 70)

    # Initialize suite
    suite = BenchmarkSuite(output_dir=Path("./benchmarks/example1"))

    # Create a retrieval function (replace with your actual implementation)
    def my_retrieval_fn(query: str):
        """Your RAG retrieval implementation."""
        # This should call your actual RAG system
        # For demo purposes, using example function
        return create_example_retrieval_function()(query)

    # Run benchmark
    model_config = {"embedding_model": "all-MiniLM-L6-v2", "chunk_size": 512}

    run = suite.run_retrieval_benchmark(
        dataset_name="custom_mcts", retrieval_fn=my_retrieval_fn, model_config=model_config, k_values=[5, 10, 20]
    )

    # Print results
    print("\nResults:")
    for metric_name, metric in run.metrics.items():
        print(f"  {metric_name}: {metric.value:.4f}")

    # Generate report
    suite.generate_report(run, output_format="markdown", output_file=suite.output_dir / "report.md")

    print(f"\nReport saved to: {suite.output_dir / 'report.md'}")


# =============================================================================
# Example 2: Reasoning Quality Evaluation
# =============================================================================


def example_reasoning_evaluation():
    """Example: Evaluate reasoning quality."""
    print("\n" + "=" * 70)
    print("Example 2: Reasoning Quality Evaluation")
    print("=" * 70)

    suite = BenchmarkSuite(output_dir=Path("./benchmarks/example2"))

    # Create reasoning function
    def my_reasoning_fn(problem: str):
        """Your reasoning implementation."""
        # Replace with actual reasoning system
        return create_example_reasoning_function()(problem)

    # Run benchmark
    model_config = {"model": "gpt-4", "temperature": 0.7}

    run = suite.run_reasoning_benchmark(
        dataset_name="gsm8k_subset", reasoning_fn=my_reasoning_fn, model_config=model_config, use_llm_judge=False
    )

    # Print results
    print("\nResults:")
    for metric_name, metric in run.metrics.items():
        ci = metric.confidence_interval
        ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        print(f"  {metric_name}: {metric.value:.4f}{ci_str}")


# =============================================================================
# Example 3: A/B Testing Between Models
# =============================================================================


def example_ab_testing():
    """Example: A/B test two different retrieval strategies."""
    print("\n" + "=" * 70)
    print("Example 3: A/B Testing Between Models")
    print("=" * 70)

    suite = BenchmarkSuite(output_dir=Path("./benchmarks/ab_test"))

    # Model A: Baseline
    def retrieval_model_a(query: str):
        """Baseline retrieval model."""
        return create_example_retrieval_function()(query)

    # Model B: Experimental
    def retrieval_model_b(query: str):
        """Experimental retrieval model with improved ranking."""
        result = create_example_retrieval_function()(query)
        # Simulate improved ranking by reordering
        if isinstance(result, dict) and "doc_ids" in result:
            doc_ids = result["doc_ids"]
            scores = result["scores"]
            # Reverse to simulate different ranking
            result["doc_ids"] = doc_ids[::-1]
            result["scores"] = scores[::-1]
        return result

    # Run both benchmarks
    run_a = suite.run_retrieval_benchmark(
        dataset_name="custom_mcts",
        retrieval_fn=retrieval_model_a,
        model_config={"name": "baseline", "version": "1.0"},
    )

    run_b = suite.run_retrieval_benchmark(
        dataset_name="custom_mcts",
        retrieval_fn=retrieval_model_b,
        model_config={"name": "experimental", "version": "2.0"},
    )

    # Compare results
    comparison = suite.compare_runs(
        baseline_run_id=run_a.timestamp,
        comparison_run_ids=[run_b.timestamp],
        output_file=suite.output_dir / "comparison.json",
    )

    print("\nComparison Results:")
    print(f"Baseline: {run_a.timestamp}")
    print(f"Experimental: {run_b.timestamp}")
    print("\nRecommendations:")
    for rec in comparison.recommendations:
        print(f"  - {rec}")


# =============================================================================
# Example 4: Custom Metrics
# =============================================================================


def example_custom_retrieval_evaluation():
    """Example: Evaluate retrieval with custom ground truth."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Retrieval Evaluation")
    print("=" * 70)

    from training.benchmark_suite import RetrievalMetrics

    # Create custom retrieval results
    results = [
        RetrievalResult(
            query="What is Monte Carlo Tree Search?",
            retrieved_docs=["doc1", "doc2", "doc5", "doc8"],
            relevance_scores=[0.95, 0.87, 0.65, 0.45],
            ground_truth_relevant=["doc1", "doc2", "doc3"],
            ground_truth_rankings={"doc1": 0, "doc2": 1, "doc3": 2},
        ),
        RetrievalResult(
            query="Explain multi-agent systems",
            retrieved_docs=["doc3", "doc4", "doc6"],
            relevance_scores=[0.92, 0.85, 0.70],
            ground_truth_relevant=["doc3", "doc4"],
            ground_truth_rankings={"doc3": 0, "doc4": 1},
        ),
    ]

    # Compute metrics manually
    print("\nMetrics for custom retrieval results:")

    for i, result in enumerate(results):
        print(f"\nQuery {i + 1}: {result.query}")
        print(f"  nDCG@10: {RetrievalMetrics.ndcg_at_k(result, k=10):.4f}")
        print(f"  Recall@10: {RetrievalMetrics.recall_at_k(result, k=10):.4f}")
        print(f"  Precision@3: {RetrievalMetrics.precision_at_k(result, k=3):.4f}")
        print(f"  MRR: {RetrievalMetrics.mean_reciprocal_rank(result):.4f}")

    # Compute MAP across all queries
    map_score = RetrievalMetrics.mean_average_precision(results)
    print(f"\nMAP across all queries: {map_score:.4f}")


# =============================================================================
# Example 5: Integration with LangSmith and W&B
# =============================================================================


def example_integration_tracking():
    """Example: Use LangSmith and W&B for tracking."""
    print("\n" + "=" * 70)
    print("Example 5: Integration with LangSmith and W&B")
    print("=" * 70)

    config = {
        "use_wandb": True,
        "wandb_project": "rag-benchmarks-demo",
        "wandb_entity": None,  # Set via WANDB_ENTITY env var
    }

    suite = BenchmarkSuite(config=config, output_dir=Path("./benchmarks/tracked"))

    # Initialize integrations
    # This requires LANGSMITH_API_KEY and WANDB_API_KEY environment variables
    suite.initialize_integrations()

    # Run benchmark
    retrieval_fn = create_example_retrieval_function()
    model_config = {"model": "tracked-model-v1", "embedding_dim": 384}

    suite.run_retrieval_benchmark(dataset_name="custom_mcts", retrieval_fn=retrieval_fn, model_config=model_config)

    # Results are automatically logged to W&B and LangSmith
    print("\nBenchmark complete!")
    print("Results logged to:")
    if suite.wandb_run:
        print(f"  W&B: {suite.wandb_run.url}")
    if suite.langsmith_client:
        print("  LangSmith: Project tracked")


# =============================================================================
# Example 6: CI/CD Integration
# =============================================================================


def example_cicd_integration():
    """Example: Use benchmark suite in CI/CD pipeline."""
    print("\n" + "=" * 70)
    print("Example 6: CI/CD Integration")
    print("=" * 70)

    suite = BenchmarkSuite(output_dir=Path("./benchmarks/cicd"))

    # Define quality thresholds
    THRESHOLDS = {"nDCG@10": 0.70, "Recall@100": 0.85, "MRR": 0.75}

    # Run benchmark
    retrieval_fn = create_example_retrieval_function()
    run = suite.run_retrieval_benchmark(dataset_name="custom_mcts", retrieval_fn=retrieval_fn, model_config={})

    # Check thresholds
    all_passed = True
    print("\nQuality Gate Checks:")

    for metric_name, threshold in THRESHOLDS.items():
        if metric_name in run.metrics:
            value = run.metrics[metric_name].value
            passed = value >= threshold
            status = "PASS" if passed else "FAIL"
            print(f"  {metric_name}: {value:.4f} (threshold: {threshold:.4f}) [{status}]")

            if not passed:
                all_passed = False
        else:
            print(f"  {metric_name}: NOT MEASURED")
            all_passed = False

    if all_passed:
        print("\nAll quality gates passed! Ready for deployment.")
        return 0
    else:
        print("\nSome quality gates failed. Blocking deployment.")
        return 1


# =============================================================================
# Example 7: Visualizations
# =============================================================================


def example_visualizations():
    """Example: Generate visualizations."""
    print("\n" + "=" * 70)
    print("Example 7: Generating Visualizations")
    print("=" * 70)

    suite = BenchmarkSuite(output_dir=Path("./benchmarks/viz"))

    # Run multiple benchmarks
    retrieval_fn = create_example_retrieval_function()

    runs = []
    for i in range(3):
        run = suite.run_retrieval_benchmark(
            dataset_name="custom_mcts",
            retrieval_fn=retrieval_fn,
            model_config={"version": f"v{i+1}", "iteration": i},
        )
        runs.append(run)

    # Generate visualizations
    plot_files = suite.visualize_results(runs=runs)

    print("\nGenerated visualizations:")
    for plot_name, plot_path in plot_files.items():
        print(f"  {plot_name}: {plot_path}")


# =============================================================================
# Example 8: Export to Different Formats
# =============================================================================


def example_export_formats():
    """Example: Export results to different formats."""
    print("\n" + "=" * 70)
    print("Example 8: Export to Different Formats")
    print("=" * 70)

    suite = BenchmarkSuite(output_dir=Path("./benchmarks/export"))

    # Run benchmark
    retrieval_fn = create_example_retrieval_function()
    run = suite.run_retrieval_benchmark(dataset_name="custom_mcts", retrieval_fn=retrieval_fn, model_config={})

    # Export to different formats
    output_dir = suite.output_dir / "reports"

    print("\nExporting results...")

    # JSON
    suite.generate_report(run, output_format="json", output_file=output_dir / "report.json")
    print(f"  JSON: {output_dir / 'report.json'}")

    # Markdown
    suite.generate_report(run, output_format="markdown", output_file=output_dir / "report.md")
    print(f"  Markdown: {output_dir / 'report.md'}")

    # HTML
    suite.generate_report(run, output_format="html", output_file=output_dir / "report.html")
    print(f"  HTML: {output_dir / 'report.html'}")

    # CSV
    suite.export_to_csv(run, output_dir / "metrics.csv")
    print(f"  CSV: {output_dir / 'metrics.csv'}")


# =============================================================================
# Main Demo Runner
# =============================================================================


def run_all_examples():
    """Run all examples."""
    logging.basicConfig(level=logging.WARNING)  # Suppress debug logs

    print("\n" + "=" * 70)
    print("BENCHMARK SUITE EXAMPLES")
    print("=" * 70)

    try:
        example_basic_retrieval_benchmark()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_reasoning_evaluation()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_ab_testing()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_custom_retrieval_evaluation()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    try:
        example_export_formats()
    except Exception as e:
        print(f"Example 8 failed: {e}")

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
