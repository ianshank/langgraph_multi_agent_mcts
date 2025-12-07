"""
Multi-Agent Ensemble for Chess Move Selection.

Integrates all available agents:
- HRM Agent: Strategic planning and opening book
- TRM Agent: Tactical move refinement
- MCTS: Deep search for best move
- Symbolic Agent: Rule-based evaluation and constraints

Captures learning from each run for dashboard analytics.

Best Practices 2025:
- Async-first design
- Comprehensive logging
- Configurable weights
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .chess_state import ChessState, ChessConfig


class AgentType(Enum):
    """Available agent types for chess."""
    HRM = auto()       # Hierarchical Reasoning Module
    TRM = auto()       # Tiny Recursive Model
    MCTS = auto()      # Monte Carlo Tree Search
    SYMBOLIC = auto()  # Neuro-Symbolic reasoning
    HYBRID = auto()    # LLM + Neural hybrid


@dataclass
class AgentResult:
    """Result from an individual agent."""
    agent_type: AgentType
    recommended_move: str
    confidence: float
    evaluation: float
    reasoning: str
    time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for the chess ensemble."""

    # Agent weights (sum to 1.0)
    hrm_weight: float = field(
        default_factory=lambda: float(os.getenv("CHESS_HRM_WEIGHT", "0.25"))
    )
    trm_weight: float = field(
        default_factory=lambda: float(os.getenv("CHESS_TRM_WEIGHT", "0.25"))
    )
    mcts_weight: float = field(
        default_factory=lambda: float(os.getenv("CHESS_MCTS_WEIGHT", "0.35"))
    )
    symbolic_weight: float = field(
        default_factory=lambda: float(os.getenv("CHESS_SYMBOLIC_WEIGHT", "0.15"))
    )

    # MCTS parameters
    mcts_simulations: int = field(
        default_factory=lambda: int(os.getenv("CHESS_MCTS_SIMULATIONS", "100"))
    )
    mcts_exploration: float = field(
        default_factory=lambda: float(os.getenv("CHESS_MCTS_EXPLORATION", "1.4"))
    )

    # Timeout settings
    agent_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("CHESS_AGENT_TIMEOUT_MS", "5000"))
    )

    # Consensus threshold
    consensus_threshold: float = field(
        default_factory=lambda: float(os.getenv("CHESS_CONSENSUS_THRESHOLD", "0.6"))
    )

    # Learning capture
    capture_learning: bool = True
    learning_db_path: str = field(
        default_factory=lambda: os.getenv(
            "CHESS_LEARNING_DB", "./chess_learning.jsonl"
        )
    )

    def __post_init__(self):
        """Normalize weights."""
        total = self.hrm_weight + self.trm_weight + self.mcts_weight + self.symbolic_weight
        if total > 0:
            self.hrm_weight /= total
            self.trm_weight /= total
            self.mcts_weight /= total
            self.symbolic_weight /= total


@dataclass
class LearningRecord:
    """Record of a single move decision for learning."""
    timestamp: str
    game_id: str
    move_number: int
    fen_before: str
    fen_after: str
    selected_move: str
    agent_results: list[dict[str, Any]]
    ensemble_confidence: float
    consensus_achieved: bool
    time_to_decide_ms: float
    game_phase: str
    evaluation_before: float
    evaluation_after: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "game_id": self.game_id,
            "move_number": self.move_number,
            "fen_before": self.fen_before,
            "fen_after": self.fen_after,
            "selected_move": self.selected_move,
            "agent_results": self.agent_results,
            "ensemble_confidence": self.ensemble_confidence,
            "consensus_achieved": self.consensus_achieved,
            "time_to_decide_ms": self.time_to_decide_ms,
            "game_phase": self.game_phase,
            "evaluation_before": self.evaluation_before,
            "evaluation_after": self.evaluation_after,
            "metadata": self.metadata,
        }


class ChessEnsemble:
    """
    Multi-agent ensemble for chess move selection.

    Coordinates HRM, TRM, MCTS, and Symbolic agents to select
    the best move through weighted voting and consensus.
    """

    def __init__(
        self,
        config: EnsembleConfig | None = None,
        game_id: str | None = None,
    ):
        self.config = config or EnsembleConfig()
        self.game_id = game_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.learning_records: list[LearningRecord] = []
        self.move_count = 0

        # Initialize agent instances (lazy loading)
        self._hrm_agent = None
        self._trm_agent = None
        self._mcts_engine = None
        self._symbolic_agent = None

    async def select_move(
        self,
        state: ChessState,
        time_limit_ms: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Select best move using all agents in ensemble.

        Args:
            state: Current chess state
            time_limit_ms: Optional time limit override

        Returns:
            Tuple of (selected_move, metadata)
        """
        start_time = time.perf_counter()
        timeout = (time_limit_ms or self.config.agent_timeout_ms) / 1000.0

        legal_moves = state.get_legal_actions()
        if not legal_moves:
            return "", {"error": "No legal moves available"}

        if len(legal_moves) == 1:
            # Only one legal move - no need for analysis
            return legal_moves[0], {"forced_move": True}

        # Run all agents in parallel
        agent_tasks = [
            self._run_hrm_agent(state, timeout),
            self._run_trm_agent(state, timeout),
            self._run_mcts_agent(state, timeout),
            self._run_symbolic_agent(state, timeout),
        ]

        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Process results
        agent_results: list[AgentResult] = []
        for result in results:
            if isinstance(result, AgentResult):
                agent_results.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Agent failed: {result}")

        # Aggregate votes
        selected_move, confidence, consensus = self._aggregate_votes(
            agent_results, legal_moves
        )

        # Calculate timing
        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Capture learning if enabled
        if self.config.capture_learning:
            self._capture_learning(
                state, selected_move, agent_results,
                confidence, consensus, total_time_ms
            )

        self.move_count += 1

        metadata = {
            "selected_move": selected_move,
            "confidence": confidence,
            "consensus_achieved": consensus,
            "time_ms": total_time_ms,
            "agent_count": len(agent_results),
            "agent_results": [
                {
                    "agent": r.agent_type.name,
                    "move": r.recommended_move,
                    "confidence": r.confidence,
                    "time_ms": r.time_ms,
                }
                for r in agent_results
            ],
            "game_phase": state.get_phase(),
            "evaluation": state.evaluate(),
        }

        return selected_move, metadata

    async def _run_hrm_agent(
        self,
        state: ChessState,
        timeout: float,
    ) -> AgentResult:
        """
        Run HRM agent for strategic move selection.

        HRM focuses on:
        - Opening principles
        - Piece coordination
        - Long-term planning
        """
        start = time.perf_counter()

        try:
            # Simplified HRM logic for demo
            move, confidence, reasoning = self._hrm_evaluate(state)

            return AgentResult(
                agent_type=AgentType.HRM,
                recommended_move=move,
                confidence=confidence,
                evaluation=state.evaluate(),
                reasoning=reasoning,
                time_ms=(time.perf_counter() - start) * 1000,
                metadata={"phase": state.get_phase()},
            )
        except Exception as e:
            logger.error(f"HRM agent error: {e}")
            raise

    def _hrm_evaluate(self, state: ChessState) -> tuple[str, float, str]:
        """HRM evaluation logic."""
        legal_moves = state.get_legal_actions()
        phase = state.get_phase()

        best_move = legal_moves[0]
        best_score = -float('inf')
        reasoning = ""

        for move_uci in legal_moves:
            score = 0.0
            reasons = []

            # Parse move
            move = chess.Move.from_uci(move_uci)
            piece = state.board.piece_at(move.from_square)

            if piece is None:
                continue

            # Opening principles
            if phase == "opening":
                # Develop minor pieces
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    to_rank = move.to_square // 8
                    if (piece.color and to_rank >= 2) or (not piece.color and to_rank <= 5):
                        score += 0.3
                        reasons.append("piece development")

                # Control center
                center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
                if move.to_square in center_squares:
                    score += 0.4
                    reasons.append("center control")

                # Castle early
                if state.board.has_castling_rights(piece.color):
                    if move.uci() in ["e1g1", "e1c1", "e8g8", "e8c8"]:
                        score += 0.5
                        reasons.append("castling")

            # Middlegame principles
            elif phase == "middlegame":
                # Piece activity
                new_state = state.apply_action(move_uci)
                mobility_diff = len(new_state.get_legal_actions()) - len(legal_moves)
                score += mobility_diff * 0.02

                # Attack coordination
                if state.board.is_attacked_by(piece.color, move.to_square):
                    score += 0.1
                    reasons.append("supported attack")

            # Endgame principles
            else:
                # King activity
                if piece.piece_type == chess.KING:
                    to_rank = move.to_square // 8
                    to_file = move.to_square % 8
                    center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
                    score += (7 - center_dist) / 7 * 0.3
                    reasons.append("king activation")

                # Pawn promotion
                if piece.piece_type == chess.PAWN:
                    to_rank = move.to_square // 8
                    if (piece.color and to_rank >= 5) or (not piece.color and to_rank <= 2):
                        score += 0.4
                        reasons.append("pawn advance")

            # Capture bonus
            if state.board.is_capture(move):
                captured = state.board.piece_at(move.to_square)
                if captured:
                    from .chess_state import PIECE_VALUES
                    score += PIECE_VALUES.get(captured.piece_type, 0) / 1000.0
                    reasons.append("capture")

            if score > best_score:
                best_score = score
                best_move = move_uci
                reasoning = f"HRM: {', '.join(reasons) if reasons else 'positional'}"

        confidence = min(0.9, 0.5 + best_score / 2)
        return best_move, confidence, reasoning

    async def _run_trm_agent(
        self,
        state: ChessState,
        timeout: float,
    ) -> AgentResult:
        """
        Run TRM agent for tactical move refinement.

        TRM focuses on:
        - Tactics detection
        - Move refinement
        - Short-term calculation
        """
        start = time.perf_counter()

        try:
            move, confidence, reasoning = self._trm_evaluate(state)

            return AgentResult(
                agent_type=AgentType.TRM,
                recommended_move=move,
                confidence=confidence,
                evaluation=state.evaluate(),
                reasoning=reasoning,
                time_ms=(time.perf_counter() - start) * 1000,
                metadata={"threats": state.get_threats()},
            )
        except Exception as e:
            logger.error(f"TRM agent error: {e}")
            raise

    def _trm_evaluate(self, state: ChessState) -> tuple[str, float, str]:
        """TRM evaluation with tactical focus."""
        legal_moves = state.get_legal_actions()

        best_move = legal_moves[0]
        best_score = -float('inf')
        reasoning = ""

        for move_uci in legal_moves:
            score = 0.0
            reasons = []
            move = chess.Move.from_uci(move_uci)

            # Check bonus (strong tactical move)
            state.board.push(move)
            if state.board.is_check():
                score += 0.5
                reasons.append("check")

                # Checkmate is highest priority
                if state.board.is_checkmate():
                    state.board.pop()
                    return move_uci, 1.0, "TRM: checkmate!"

            state.board.pop()

            # Capture with material gain
            if state.board.is_capture(move):
                captured = state.board.piece_at(move.to_square)
                attacker = state.board.piece_at(move.from_square)
                if captured and attacker:
                    from .chess_state import PIECE_VALUES
                    material_gain = (
                        PIECE_VALUES.get(captured.piece_type, 0) -
                        PIECE_VALUES.get(attacker.piece_type, 0)
                    )
                    if material_gain > 0:
                        score += material_gain / 500.0
                        reasons.append(f"winning capture (+{material_gain})")
                    elif material_gain == 0:
                        score += 0.1
                        reasons.append("equal trade")

            # Fork detection (simplified)
            state.board.push(move)
            attacks = 0
            for sq in chess.SQUARES:
                if state.board.is_attacked_by(state.board.turn, sq):
                    piece = state.board.piece_at(sq)
                    if piece and piece.color != state.board.turn:
                        attacks += 1
            if attacks >= 2:
                score += 0.3
                reasons.append("multiple attacks")
            state.board.pop()

            # Pin/skewer detection (simplified)
            piece = state.board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                # Check if move creates pin
                state.board.push(move)
                for sq in chess.SQUARES:
                    target = state.board.piece_at(sq)
                    if target and target.color != state.board.turn:
                        if state.board.is_pinned(not state.board.turn, sq):
                            score += 0.2
                            reasons.append("creates pin")
                            break
                state.board.pop()

            if score > best_score:
                best_score = score
                best_move = move_uci
                reasoning = f"TRM: {', '.join(reasons) if reasons else 'solid'}"

        confidence = min(0.9, 0.5 + best_score)
        return best_move, confidence, reasoning

    async def _run_mcts_agent(
        self,
        state: ChessState,
        timeout: float,
    ) -> AgentResult:
        """
        Run MCTS for deep search.

        Uses configurable number of simulations.
        """
        start = time.perf_counter()

        try:
            move, confidence, visits = self._mcts_search(state)

            return AgentResult(
                agent_type=AgentType.MCTS,
                recommended_move=move,
                confidence=confidence,
                evaluation=state.evaluate(),
                reasoning=f"MCTS: {visits} simulations",
                time_ms=(time.perf_counter() - start) * 1000,
                metadata={"simulations": visits},
            )
        except Exception as e:
            logger.error(f"MCTS agent error: {e}")
            raise

    def _mcts_search(self, state: ChessState) -> tuple[str, float, int]:
        """Simple MCTS search for chess."""
        legal_moves = state.get_legal_actions()

        if len(legal_moves) == 1:
            return legal_moves[0], 1.0, 1

        # Statistics for each move
        visits = {move: 0 for move in legal_moves}
        wins = {move: 0.0 for move in legal_moves}

        num_simulations = self.config.mcts_simulations

        for _ in range(num_simulations):
            # Selection: UCB1
            total_visits = sum(visits.values()) + 1

            best_ucb = -float('inf')
            selected_move = legal_moves[0]

            for move in legal_moves:
                if visits[move] == 0:
                    ucb = float('inf')
                else:
                    exploitation = wins[move] / visits[move]
                    exploration = self.config.mcts_exploration * (
                        (2 * (total_visits) ** 0.5) / (visits[move] + 1)
                    ) ** 0.5
                    ucb = exploitation + exploration

                if ucb > best_ucb:
                    best_ucb = ucb
                    selected_move = move

            # Simulation: Random rollout with evaluation
            new_state = state.apply_action(selected_move)
            value = self._rollout(new_state, max_depth=10)

            # Backpropagation
            visits[selected_move] += 1
            wins[selected_move] += (value + 1) / 2  # Normalize to [0, 1]

        # Select best move by visit count
        best_move = max(legal_moves, key=lambda m: visits[m])
        total_visits = sum(visits.values())
        confidence = visits[best_move] / total_visits if total_visits > 0 else 0.5

        return best_move, confidence, num_simulations

    def _rollout(self, state: ChessState, max_depth: int) -> float:
        """Random rollout with evaluation cutoff."""
        current = state
        depth = 0

        while not current.is_terminal() and depth < max_depth:
            legal = current.get_legal_actions()
            if not legal:
                break

            # Semi-random: bias toward captures
            import random
            captures = [m for m in legal if current.board.is_capture(chess.Move.from_uci(m))]
            if captures and random.random() < 0.3:
                move = random.choice(captures)
            else:
                move = random.choice(legal)

            current = current.apply_action(move)
            depth += 1

        return current.evaluate()

    async def _run_symbolic_agent(
        self,
        state: ChessState,
        timeout: float,
    ) -> AgentResult:
        """
        Run symbolic reasoning agent.

        Uses rule-based evaluation and constraint checking.
        """
        start = time.perf_counter()

        try:
            move, confidence, reasoning = self._symbolic_evaluate(state)

            return AgentResult(
                agent_type=AgentType.SYMBOLIC,
                recommended_move=move,
                confidence=confidence,
                evaluation=state.evaluate(),
                reasoning=reasoning,
                time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Symbolic agent error: {e}")
            raise

    def _symbolic_evaluate(self, state: ChessState) -> tuple[str, float, str]:
        """Symbolic/rule-based evaluation."""
        legal_moves = state.get_legal_actions()
        phase = state.get_phase()

        # Apply rule-based constraints
        rules_applied = []
        move_scores = {move: 0.0 for move in legal_moves}

        for move_uci in legal_moves:
            move = chess.Move.from_uci(move_uci)
            piece = state.board.piece_at(move.from_square)

            if piece is None:
                continue

            # Rule: Don't move the same piece twice in opening
            if phase == "opening" and len(state.move_history) < 10:
                if len(state.move_history) >= 2:
                    last_move = chess.Move.from_uci(state.move_history[-2])
                    if last_move.to_square == move.from_square:
                        move_scores[move_uci] -= 0.2
                        if "avoid repetition" not in rules_applied:
                            rules_applied.append("avoid repetition")

            # Rule: Don't bring queen out early
            if phase == "opening" and piece.piece_type == chess.QUEEN:
                if len(state.move_history) < 6:
                    move_scores[move_uci] -= 0.3
                    if "queen out early" not in rules_applied:
                        rules_applied.append("queen out early")

            # Rule: Castle when possible
            if move_uci in ["e1g1", "e1c1", "e8g8", "e8c8"]:
                move_scores[move_uci] += 0.4
                rules_applied.append("castling available")

            # Rule: Don't move king unless necessary (before castling)
            if piece.piece_type == chess.KING and phase != "endgame":
                if state.board.has_castling_rights(piece.color):
                    move_scores[move_uci] -= 0.2
                    if "preserve castling" not in rules_applied:
                        rules_applied.append("preserve castling")

            # Rule: Prioritize development in opening
            if phase == "opening":
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    home_rank = 0 if piece.color else 7
                    if move.from_square // 8 == home_rank:
                        move_scores[move_uci] += 0.2
                        if "develop pieces" not in rules_applied:
                            rules_applied.append("develop pieces")

            # Rule: Trade when ahead in material
            material = self._count_material(state.board)
            if state.board.is_capture(move):
                if (state.board.turn and material > 200) or (not state.board.turn and material < -200):
                    move_scores[move_uci] += 0.15
                    if "trade when ahead" not in rules_applied:
                        rules_applied.append("trade when ahead")

        # Select best move by rule score
        best_move = max(legal_moves, key=lambda m: move_scores[m])
        best_score = move_scores[best_move]

        confidence = min(0.9, 0.5 + best_score)
        reasoning = f"Symbolic: {', '.join(rules_applied) if rules_applied else 'default rules'}"

        return best_move, confidence, reasoning

    def _count_material(self, board: Any) -> int:
        """Count material balance."""
        from .chess_state import PIECE_VALUES
        total = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white = len(board.pieces(piece_type, chess.WHITE))
            black = len(board.pieces(piece_type, chess.BLACK))
            total += (white - black) * PIECE_VALUES.get(piece_type, 0)
        return total

    def _aggregate_votes(
        self,
        results: list[AgentResult],
        legal_moves: list[str],
    ) -> tuple[str, float, bool]:
        """
        Aggregate agent votes to select best move.

        Uses weighted voting with consensus detection.
        """
        if not results:
            return legal_moves[0], 0.0, False

        # Weight by agent type
        weights = {
            AgentType.HRM: self.config.hrm_weight,
            AgentType.TRM: self.config.trm_weight,
            AgentType.MCTS: self.config.mcts_weight,
            AgentType.SYMBOLIC: self.config.symbolic_weight,
        }

        # Accumulate votes for each move
        move_votes: dict[str, float] = {}

        for result in results:
            move = result.recommended_move
            if move not in legal_moves:
                continue

            weight = weights.get(result.agent_type, 0.1)
            vote = weight * result.confidence

            move_votes[move] = move_votes.get(move, 0.0) + vote

        if not move_votes:
            return legal_moves[0], 0.0, False

        # Select move with highest weighted votes
        best_move = max(move_votes.keys(), key=lambda m: move_votes[m])
        total_votes = sum(move_votes.values())
        confidence = move_votes[best_move] / total_votes if total_votes > 0 else 0.0

        # Check for consensus
        same_move_count = sum(1 for r in results if r.recommended_move == best_move)
        consensus = (same_move_count / len(results)) >= self.config.consensus_threshold

        return best_move, confidence, consensus

    def _capture_learning(
        self,
        state: ChessState,
        selected_move: str,
        agent_results: list[AgentResult],
        confidence: float,
        consensus: bool,
        time_ms: float,
    ) -> None:
        """Capture learning record for this decision."""
        new_state = state.apply_action(selected_move)

        record = LearningRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            game_id=self.game_id,
            move_number=self.move_count + 1,
            fen_before=state.get_fen(),
            fen_after=new_state.get_fen(),
            selected_move=selected_move,
            agent_results=[
                {
                    "agent": r.agent_type.name,
                    "move": r.recommended_move,
                    "confidence": r.confidence,
                    "evaluation": r.evaluation,
                    "reasoning": r.reasoning,
                    "time_ms": r.time_ms,
                }
                for r in agent_results
            ],
            ensemble_confidence=confidence,
            consensus_achieved=consensus,
            time_to_decide_ms=time_ms,
            game_phase=state.get_phase(),
            evaluation_before=state.evaluate(),
            evaluation_after=new_state.evaluate(),
        )

        self.learning_records.append(record)

        # Append to learning database file
        try:
            with open(self.config.learning_db_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save learning record: {e}")

    def get_learning_summary(self) -> dict[str, Any]:
        """Get summary of learning from this game."""
        if not self.learning_records:
            return {"game_id": self.game_id, "moves": 0}

        agent_agreement = {}
        agent_times = {}
        phase_distribution = {}

        for record in self.learning_records:
            # Track agent agreement
            for ar in record.agent_results:
                agent = ar["agent"]
                agreed = ar["move"] == record.selected_move
                if agent not in agent_agreement:
                    agent_agreement[agent] = {"agreed": 0, "total": 0}
                agent_agreement[agent]["total"] += 1
                if agreed:
                    agent_agreement[agent]["agreed"] += 1

                # Track agent times
                if agent not in agent_times:
                    agent_times[agent] = []
                agent_times[agent].append(ar["time_ms"])

            # Phase distribution
            phase = record.game_phase
            phase_distribution[phase] = phase_distribution.get(phase, 0) + 1

        return {
            "game_id": self.game_id,
            "total_moves": len(self.learning_records),
            "avg_confidence": sum(r.ensemble_confidence for r in self.learning_records) / len(self.learning_records),
            "consensus_rate": sum(1 for r in self.learning_records if r.consensus_achieved) / len(self.learning_records),
            "avg_decision_time_ms": sum(r.time_to_decide_ms for r in self.learning_records) / len(self.learning_records),
            "agent_agreement_rate": {
                agent: data["agreed"] / data["total"] if data["total"] > 0 else 0
                for agent, data in agent_agreement.items()
            },
            "agent_avg_time_ms": {
                agent: sum(times) / len(times) if times else 0
                for agent, times in agent_times.items()
            },
            "phase_distribution": phase_distribution,
            "evaluation_trend": [r.evaluation_after for r in self.learning_records],
        }
