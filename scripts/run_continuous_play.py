#!/usr/bin/env python3
"""
Continuous Play Session Runner.

Executes a continuous self-play learning session with live monitoring.
Configurable via environment variables or command line arguments.

Usage:
    python scripts/run_continuous_play.py --duration 20 --max-games 100

Environment Variables:
    SESSION_DURATION_MIN: Session duration in minutes (default: 20)
    MAX_GAMES: Maximum number of games (default: 100)
    CHESS_PRESET: Chess configuration preset (default: small)
    ENABLE_PROMETHEUS: Enable Prometheus metrics (default: false)
    REPORT_DIR: Directory for reports (default: ./reports)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.continuous_play_config import ContinuousPlayConfig
from src.training.continuous_play_orchestrator import ContinuousPlayOrchestrator


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with timestamps and levels."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a continuous self-play learning session",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=int(os.environ.get("SESSION_DURATION_MIN", "20")),
        help="Session duration in minutes",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=int(os.environ.get("MAX_GAMES", "100")),
        help="Maximum number of games to play",
    )
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large"],
        default=os.environ.get("CHESS_PRESET", "small"),
        help="Chess configuration preset",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=os.environ.get("DEVICE", "cuda"),
        help="Compute device",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=os.environ.get("REPORT_DIR", "./reports"),
        help="Directory for output reports",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "--enable-prometheus",
        action="store_true",
        default=os.environ.get("ENABLE_PROMETHEUS", "false").lower() == "true",
        help="Enable Prometheus metrics export",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("CHECKPOINT_DIR", "./checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=os.environ.get("LOAD_CHECKPOINT", None),
        help="Path to checkpoint directory to resume from",
    )

    return parser.parse_args()


from collections.abc import Callable


def create_progress_printer(start_time: datetime) -> Callable[[int, int, object], None]:
    """Create a progress callback that prints to console."""
    last_print_time = [start_time]

    def progress_callback(game_num: int, max_games: int, scorecard) -> None:
        now = datetime.now()
        # Print every 5 seconds at most
        if (now - last_print_time[0]).total_seconds() >= 5:
            elapsed = (now - start_time).total_seconds() / 60
            print(
                f"\n{'=' * 60}\n"
                f"Game {game_num}/{max_games} | "
                f"Elapsed: {elapsed:.1f} min | "
                f"Elo: {scorecard.elo_estimate:.0f}\n"
                f"W: {scorecard.white_wins} | B: {scorecard.black_wins} | D: {scorecard.draws} | "
                f"Win Rate: {scorecard.win_rate:.1%}\n"
                f"Positions Learned: {scorecard.total_positions_learned} | "
                f"Last Loss: {scorecard.last_loss:.4f}\n"
                f"{'=' * 60}",
                flush=True,
            )
            last_print_time[0] = now

    return progress_callback


async def run_session(args: argparse.Namespace) -> int:
    """Run the continuous play session."""
    # Create timestamped report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("CONTINUOUS PLAY & LEARNING SESSION")
    print(f"{'=' * 70}")
    print(f"Duration:    {args.duration} minutes")
    print(f"Max Games:   {args.max_games}")
    print(f"Preset:      {args.preset}")
    print(f"Device:      {args.device}")
    print(f"Report Dir:  {report_dir}")
    print(f"{'=' * 70}\n")

    # Configure via environment (picked up by config loader)
    os.environ["SESSION_DURATION_MIN"] = str(args.duration)
    os.environ["MAX_GAMES"] = str(args.max_games)
    os.environ["CHESS_PRESET"] = args.preset
    os.environ["DEVICE"] = args.device
    os.environ["REPORT_DIR"] = str(report_dir)
    os.environ["GENERATE_HTML_REPORT"] = "true"
    os.environ["GENERATE_JSON_REPORT"] = "true"
    os.environ["ENABLE_PROMETHEUS"] = str(args.enable_prometheus).lower()
    os.environ["CHECKPOINT_DIR"] = str(args.checkpoint_dir)
    if args.resume:
        os.environ["LOAD_CHECKPOINT"] = str(args.resume)

    # Create config and orchestrator
    config = ContinuousPlayConfig.from_env()
    config.metrics.report_output_dir = str(report_dir)

    orchestrator = ContinuousPlayOrchestrator(config)

    # Set up live monitoring callback
    def metrics_callback(metrics: dict) -> None:
        # Could be used for real-time dashboard updates
        pass

    orchestrator.register_metrics_callback(metrics_callback)

    try:
        print("Starting session...\n")
        start_time = datetime.now()

        # Create progress printer with start time
        progress_callback = create_progress_printer(start_time)

        result = await orchestrator.run_session(progress_callback=progress_callback)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        # Print final summary
        print(f"\n{'=' * 70}")
        print("SESSION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total Duration:    {duration:.1f} minutes")
        print(f"Total Games:       {result.total_games}")
        print(f"Games per Minute:  {result.games_per_minute:.2f}")
        print(f"{'=' * 70}")
        print(f"Final Elo:         {result.final_elo:.0f}")
        print(f"Elo Change:        {result.elo_delta:+.0f}")
        print(f"Positions Learned: {result.total_positions_learned}")
        print(f"Avg Training Loss: {result.avg_training_loss:.4f}")
        print(f"{'=' * 70}")

        if result.report_path:
            print(f"HTML Report:       {result.report_path}")
        if result.metrics_path:
            print(f"JSON Metrics:      {result.metrics_path}")

        print(f"{'=' * 70}\n")

        # Get improvement summary
        summary = orchestrator.get_improvement_summary()

        print("IMPROVEMENT SUMMARY:")
        print(f"  Elo Delta (Total):  {summary['elo_delta_total']:+.0f}")
        print(f"  Elo Delta (Recent): {summary['elo_delta_recent']:+.0f}")
        print(f"  Games Completed:    {summary['games_completed']}")
        print(f"  Overall Win Rate:   {summary['win_rate']:.1%}")
        print()

        # Success if we played at least some games
        if result.total_games > 0:
            print("Session completed successfully!")
            return 0
        else:
            print("Warning: No games were played")
            return 1

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        orchestrator.stop()
        return 130
    except Exception as e:
        logging.exception("Session failed with error: %s", e)
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    return asyncio.run(run_session(args))


if __name__ == "__main__":
    sys.exit(main())
