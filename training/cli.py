#!/usr/bin/env python3
"""
Command-line interface for Multi-Agent MCTS Training Pipeline.

Usage:
    python -m training.cli train --config training/config.yaml
    python -m training.cli evaluate --model models/hrm_checkpoint.pt
    python -m training.cli build-rag --output cache/rag_index
    python -m training.cli monitor --port 8000
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def train_command(args):
    """Run training pipeline."""
    from training.orchestrator import TrainingPipeline

    logger = logging.getLogger(__name__)
    logger.info(f"Starting training pipeline with config: {args.config}")

    pipeline = TrainingPipeline(args.config)

    if args.phase:
        # Run specific phase
        logger.info(f"Running phase: {args.phase}")
        pipeline.initialize_components()
        phase = next((p for p in pipeline.phase_manager.phases if p["name"] == args.phase), None)
        if phase:
            result = pipeline._run_phase(phase)
            logger.info(f"Phase result: {result}")
        else:
            logger.error(f"Phase {args.phase} not found")
            sys.exit(1)
    else:
        # Run full pipeline
        results = pipeline.run_full_pipeline(resume_from=args.resume)
        logger.info(f"Training completed. Total time: {results['total_time'] / 3600:.2f} hours")

        if args.output:
            with open(args.output, "w") as f:
                import json

                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")


def evaluate_command(args):
    """Run evaluation on trained models."""
    from training.data_pipeline import DABStepLoader
    from training.evaluation import DABStepBenchmark, ProductionValidator

    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating model: {args.model}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup evaluator
    benchmark = DABStepBenchmark(config["evaluation"])

    # Load test data
    dabstep_loader = DABStepLoader(config["data"]["dabstep"])
    splits = dabstep_loader.create_splits()
    test_data = splits["test"]

    # Create mock model or load real one
    if args.model == "mock":

        class MockModel:
            def predict(self, sample):
                return {
                    "answer": sample.get("expected_output", ""),
                    "steps": len(sample.get("steps", [])),
                    "reasoning": sample.get("steps", []),
                    "confidence": 0.85,
                    "consensus": 0.9,
                }

        model = MockModel()
    else:
        # Load actual trained model
        import torch

        checkpoint = torch.load(args.model, map_location="cpu")
        # Extract model type from checkpoint or default
        model_type = checkpoint.get("model_type", "hrm") if isinstance(checkpoint, dict) else "hrm"

        # Instantiate model based on type and load state dict
        if model_type == "hrm":
            from training.agent_trainer import HRMTrainer

            trainer = HRMTrainer(config["agents"]["hrm"])
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                trainer.model.load_state_dict(checkpoint["model_state_dict"])
            model = trainer.model
        elif model_type == "trm":
            from training.agent_trainer import TRMTrainer

            trainer = TRMTrainer(config["agents"]["trm"])
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                trainer.model.load_state_dict(checkpoint["model_state_dict"])
            model = trainer.model
        else:
            # Generic model loading
            logger.warning(f"Unknown model type: {model_type}, using checkpoint directly")
            model = checkpoint

    # Run evaluation
    report = benchmark.evaluate_model(model, test_data, verbose=True)

    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {report.accuracy:.2%}")
    logger.info(f"  F1 Score: {report.f1_score:.4f}")
    logger.info(f"  Avg Latency: {report.avg_latency_ms:.2f}ms")

    # Production validation
    if args.validate_production:
        validator = ProductionValidator(config["evaluation"])
        passed, checks = validator.validate_all_criteria(report)
        logger.info(f"Production Ready: {passed}")
        for check, status in checks.items():
            logger.info(f"  {check}: {'PASS' if status else 'FAIL'}")


def build_rag_command(args):
    """Build RAG index from PRIMUS dataset."""
    from training.data_pipeline import PRIMUSProcessor
    from training.rag_builder import RAGIndexManager

    logger = logging.getLogger(__name__)
    logger.info(f"Building RAG index to: {args.output}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup processor
    processor = PRIMUSProcessor(config["data"])

    # Build indices
    rag_manager = RAGIndexManager(args.config)
    stats = rag_manager.build_domain_indices(processor)

    logger.info("RAG Index Statistics:")
    for domain, domain_stats in stats.items():
        logger.info(f"  {domain}:")
        logger.info(f"    Documents: {domain_stats.total_documents}")
        logger.info(f"    Chunks: {domain_stats.total_chunks}")
        logger.info(f"    Index Size: {domain_stats.index_size_mb:.2f} MB")

    # Save indices
    rag_manager.save_all_indices()
    logger.info("RAG indices saved successfully")


def monitor_command(args):
    """Start monitoring dashboard."""
    from training.monitoring import MetricsDashboard, TrainingMonitor

    logger = logging.getLogger(__name__)
    logger.info(f"Starting monitoring dashboard on port {args.port}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    monitor = TrainingMonitor(config.get("monitoring", {}))
    dashboard = MetricsDashboard(monitor)

    # Generate static dashboard
    if args.static:
        html_path = dashboard.generate_html_report()
        logger.info(f"Static dashboard generated: {html_path}")
    else:
        logger.info("Live monitoring not yet implemented")
        logger.info("Use --static flag to generate static HTML report")


def meta_controller_command(args):
    """Train meta-controller."""
    from training.meta_controller import MetaControllerTrainer

    logger = logging.getLogger(__name__)
    logger.info("Training meta-controller")

    trainer = MetaControllerTrainer(args.config)

    if args.generate_traces:
        trainer.trace_collector.generate_synthetic_traces(args.num_traces)
        logger.info(f"Generated {args.num_traces} synthetic traces")

    # Train router
    history = trainer.train_router(num_epochs=args.epochs)

    logger.info("Training complete:")
    logger.info(f"  Final Loss: {history['loss'][-1]:.4f}")
    logger.info(f"  Final Accuracy: {history['accuracy'][-1]:.2%}")

    # Save checkpoint
    trainer.save_checkpoint()
    logger.info("Meta-controller checkpoint saved")


def integrate_command(args):
    """Integrate trained models into production system."""
    from training.integrate import ConfigurationManager, ModelIntegrator

    logger = logging.getLogger(__name__)
    logger.info(f"Integrating models from {args.models_dir}")

    integrator = ModelIntegrator(args.config)
    config_manager = ConfigurationManager(args.config)

    # Load models
    models_dir = Path(args.models_dir)
    for model_file in models_dir.glob("*.pt"):
        model_type = model_file.stem.split("_")[0]  # Extract type from filename
        integrator.load_trained_model(model_type, str(model_file))
        logger.info(f"Loaded {model_type} model")

    # Export for production
    if args.export:
        integrator.export_production_models(args.export)
        logger.info(f"Exported models to {args.export}")

    # Create production config
    if args.production_config:
        config_manager.create_production_config()
        logger.info("Created production configuration")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent MCTS Training Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full training pipeline
  python -m training.cli train --config training/config.yaml

  # Run specific training phase
  python -m training.cli train --config training/config.yaml --phase base_pretraining

  # Resume from checkpoint
  python -m training.cli train --config training/config.yaml --resume training/models/checkpoints/pipeline_state.json

  # Evaluate trained model
  python -m training.cli evaluate --model models/hrm_checkpoint.pt --config training/config.yaml

  # Build RAG index
  python -m training.cli build-rag --output cache/rag_index --config training/config.yaml

  # Train meta-controller
  python -m training.cli meta-controller --config training/config.yaml --epochs 10

  # Generate monitoring dashboard
  python -m training.cli monitor --static --config training/config.yaml
        """,
    )

    # Global arguments
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    parser.add_argument("--log-file", default=None, help="Log file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    train_parser.add_argument("--phase", default=None, help="Run specific phase")
    train_parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    train_parser.add_argument("--output", default=None, help="Output file for results")
    train_parser.set_defaults(func=train_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--model", required=True, help="Model checkpoint path or 'mock'")
    eval_parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    eval_parser.add_argument("--validate-production", action="store_true", help="Run production validation")
    eval_parser.set_defaults(func=evaluate_command)

    # Build RAG command
    rag_parser = subparsers.add_parser("build-rag", help="Build RAG index")
    rag_parser.add_argument("--output", required=True, help="Output directory for indices")
    rag_parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    rag_parser.set_defaults(func=build_rag_command)

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start monitoring dashboard")
    monitor_parser.add_argument("--port", type=int, default=8000, help="Dashboard port")
    monitor_parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    monitor_parser.add_argument("--static", action="store_true", help="Generate static HTML report")
    monitor_parser.set_defaults(func=monitor_command)

    # Meta-controller command
    mc_parser = subparsers.add_parser("meta-controller", help="Train meta-controller")
    mc_parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    mc_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    mc_parser.add_argument("--generate-traces", action="store_true", help="Generate synthetic traces")
    mc_parser.add_argument("--num-traces", type=int, default=10000, help="Number of traces to generate")
    mc_parser.set_defaults(func=meta_controller_command)

    # Integrate command
    integrate_parser = subparsers.add_parser("integrate", help="Integrate trained models")
    integrate_parser.add_argument("--models-dir", required=True, help="Directory with trained models")
    integrate_parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    integrate_parser.add_argument("--export", default=None, help="Export models to directory")
    integrate_parser.add_argument("--production-config", action="store_true", help="Create production config")
    integrate_parser.set_defaults(func=integrate_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Run command
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
