"""
Chess Web UI with Gradio.

Provides a web interface for playing chess against the AlphaZero-style
ensemble AI with real-time board visualization and move history.
"""

from __future__ import annotations

import asyncio
import html
import logging
from dataclasses import dataclass, field
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)


@dataclass
class GameSession:
    """Maintains state for a chess game session."""

    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move_history: list[str] = field(default_factory=list)
    game_over: bool = False
    result: str = ""
    player_color: str = "white"
    ai_thinking: bool = False
    last_ai_analysis: dict[str, Any] = field(default_factory=dict)

    def reset(self, player_color: str = "white") -> None:
        """Reset the game to initial state."""
        self.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.move_history = []
        self.game_over = False
        self.result = ""
        self.player_color = player_color
        self.ai_thinking = False
        self.last_ai_analysis = {}


# Global session (for simplicity - in production use proper session management)
_session = GameSession()


def get_piece_unicode(piece: str) -> str:
    """Convert piece character to Unicode chess symbol."""
    pieces = {
        "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
        "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    }
    return pieces.get(piece, "")


def fen_to_board(fen: str) -> list[list[str]]:
    """Convert FEN string to 8x8 board array."""
    board = []
    rows = fen.split()[0].split("/")

    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend([""] * int(char))
            else:
                board_row.append(char)
        board.append(board_row)

    return board


def render_board_html(fen: str, last_move: str | None = None, flipped: bool = False) -> str:
    """Render chess board as HTML with CSS styling."""
    board = fen_to_board(fen)

    # Parse last move for highlighting
    highlight_squares = set()
    if last_move and len(last_move) >= 4:
        from_sq = last_move[:2]
        to_sq = last_move[2:4]
        from_file = ord(from_sq[0]) - ord("a")
        from_rank = 8 - int(from_sq[1])
        to_file = ord(to_sq[0]) - ord("a")
        to_rank = 8 - int(to_sq[1])
        highlight_squares.add((from_rank, from_file))
        highlight_squares.add((to_rank, to_file))

    html_parts = ["""
    <style>
        .chess-board {
            display: inline-block;
            border: 3px solid #333;
            border-radius: 4px;
            font-family: 'Segoe UI Symbol', 'DejaVu Sans', sans-serif;
        }
        .chess-row {
            display: flex;
            height: 60px;
        }
        .chess-square {
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 42px;
            cursor: pointer;
            user-select: none;
        }
        .chess-square:hover {
            outline: 3px solid #007bff;
            outline-offset: -3px;
        }
        .light-square {
            background-color: #f0d9b5;
        }
        .dark-square {
            background-color: #b58863;
        }
        .highlight-square {
            background-color: #aaa23a !important;
        }
        .rank-label, .file-label {
            font-size: 12px;
            color: #666;
            font-weight: bold;
        }
        .rank-labels {
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            padding-right: 5px;
            height: 480px;
        }
        .file-labels {
            display: flex;
            justify-content: space-around;
            padding-left: 25px;
            width: 480px;
        }
        .board-container {
            display: flex;
            align-items: center;
        }
        .board-wrapper {
            display: flex;
            flex-direction: column;
        }
    </style>
    <div class="board-wrapper">
        <div class="board-container">
            <div class="rank-labels">
    """]

    # Rank labels
    ranks = list(range(8, 0, -1)) if not flipped else list(range(1, 9))
    for rank in ranks:
        html_parts.append(f'<span class="rank-label">{rank}</span>')

    html_parts.append('</div><div class="chess-board" id="chess-board">')

    # Board squares
    row_indices = range(8) if not flipped else range(7, -1, -1)
    col_indices = range(8) if not flipped else range(7, -1, -1)

    for row_idx, board_row in enumerate(row_indices):
        html_parts.append('<div class="chess-row">')
        for col_idx, board_col in enumerate(col_indices):
            is_light = (board_row + board_col) % 2 == 0
            square_class = "light-square" if is_light else "dark-square"

            if (board_row, board_col) in highlight_squares:
                square_class += " highlight-square"

            piece = board[board_row][board_col]
            piece_symbol = get_piece_unicode(piece)

            # Calculate square name for data attribute
            file_char = chr(ord("a") + board_col)
            rank_num = 8 - board_row
            square_name = f"{file_char}{rank_num}"

            html_parts.append(
                f'<div class="chess-square {square_class}" '
                f'data-square="{square_name}" data-piece="{piece}">'
                f'{piece_symbol}</div>'
            )
        html_parts.append("</div>")

    html_parts.append("</div></div>")

    # File labels
    html_parts.append('<div class="file-labels">')
    files = "abcdefgh" if not flipped else "hgfedcba"
    for f in files:
        html_parts.append(f'<span class="file-label">{f}</span>')
    html_parts.append("</div></div>")

    return "".join(html_parts)


def format_move_history(moves: list[str]) -> str:
    """Format move history as PGN-like notation."""
    if not moves:
        return "No moves yet"

    lines = []
    for i in range(0, len(moves), 2):
        move_num = i // 2 + 1
        white_move = moves[i]
        black_move = moves[i + 1] if i + 1 < len(moves) else ""
        lines.append(f"{move_num}. {white_move} {black_move}".strip())

    return "\n".join(lines)


def get_game_status(session: GameSession) -> str:
    """Get current game status message."""
    if session.game_over:
        return f"Game Over: {session.result}"

    try:
        import chess
        board = chess.Board(session.fen)

        if board.is_check():
            return "Check!"

        turn = "White" if board.turn else "Black"
        return f"{turn} to move"
    except Exception:
        return "Game in progress"


def initialize_game(player_color: str) -> tuple[str, str, str, str]:
    """Initialize a new game.

    Returns:
        (board_html, status, move_history, analysis)
    """
    global _session
    _session.reset(player_color)

    board_html = render_board_html(_session.fen, flipped=(player_color == "black"))
    status = get_game_status(_session)
    history = format_move_history(_session.move_history)

    # If player is black, AI moves first
    if player_color == "black":
        return make_ai_move_sync()

    return board_html, status, history, "New game started. Your turn!"


def validate_move(move_uci: str) -> tuple[bool, str]:
    """Validate a move in UCI format.

    Returns:
        (is_valid, error_message)
    """
    if not move_uci or len(move_uci) < 4:
        return False, "Invalid move format. Use UCI notation (e.g., e2e4)"

    try:
        import chess
        board = chess.Board(_session.fen)
        move = chess.Move.from_uci(move_uci.lower().strip())

        if move not in board.legal_moves:
            legal = [m.uci() for m in list(board.legal_moves)[:10]]
            return False, f"Illegal move. Legal moves include: {', '.join(legal)}..."

        return True, ""
    except Exception as e:
        return False, f"Error validating move: {str(e)}"


def apply_player_move(move_uci: str) -> tuple[str, str, str, str]:
    """Apply player's move and get AI response.

    Returns:
        (board_html, status, move_history, analysis)
    """
    global _session

    if _session.game_over:
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            "Game is over. Start a new game.",
        )

    # Validate move
    is_valid, error = validate_move(move_uci)
    if not is_valid:
        return (
            render_board_html(
                _session.fen,
                _session.move_history[-1] if _session.move_history else None,
                flipped=(_session.player_color == "black"),
            ),
            get_game_status(_session),
            format_move_history(_session.move_history),
            error,
        )

    # Apply move
    try:
        import chess
        board = chess.Board(_session.fen)
        move = chess.Move.from_uci(move_uci.lower().strip())
        board.push(move)

        _session.fen = board.fen()
        _session.move_history.append(move_uci.lower())

        # Check for game over
        if board.is_game_over():
            _session.game_over = True
            if board.is_checkmate():
                _session.result = "You win by checkmate!"
            elif board.is_stalemate():
                _session.result = "Draw by stalemate"
            else:
                _session.result = "Draw"

            return (
                render_board_html(
                    _session.fen,
                    move_uci,
                    flipped=(_session.player_color == "black"),
                ),
                get_game_status(_session),
                format_move_history(_session.move_history),
                _session.result,
            )

        # AI's turn
        return make_ai_move_sync()

    except Exception as e:
        logger.exception("Error applying move")
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            f"Error: {str(e)}",
        )


def make_ai_move_sync() -> tuple[str, str, str, str]:
    """Make AI move synchronously.

    Returns:
        (board_html, status, move_history, analysis)
    """
    global _session

    try:
        import chess

        board = chess.Board(_session.fen)

        if board.is_game_over():
            _session.game_over = True
            if board.is_checkmate():
                _session.result = "AI wins by checkmate!"
            else:
                _session.result = "Draw"
            return (
                render_board_html(_session.fen, flipped=(_session.player_color == "black")),
                get_game_status(_session),
                format_move_history(_session.move_history),
                _session.result,
            )

        # Try to use ensemble agent, fallback to simple move selection
        ai_move, analysis = get_ai_move(_session.fen)

        # Apply AI move
        move = chess.Move.from_uci(ai_move)
        board.push(move)

        _session.fen = board.fen()
        _session.move_history.append(ai_move)
        _session.last_ai_analysis = analysis

        # Check for game over
        if board.is_game_over():
            _session.game_over = True
            if board.is_checkmate():
                _session.result = "AI wins by checkmate!"
            elif board.is_stalemate():
                _session.result = "Draw by stalemate"
            else:
                _session.result = "Draw"

        return (
            render_board_html(
                _session.fen,
                ai_move,
                flipped=(_session.player_color == "black"),
            ),
            get_game_status(_session),
            format_move_history(_session.move_history),
            format_analysis(ai_move, analysis),
        )

    except Exception as e:
        logger.exception("Error making AI move")
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            f"AI Error: {str(e)}",
        )


def get_ai_move(fen: str) -> tuple[str, dict[str, Any]]:
    """Get AI's move for the position.

    Returns:
        (move_uci, analysis_dict)
    """
    import chess

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    # Try to use ensemble agent
    try:
        from src.games.chess import ChessEnsembleAgent, ChessGameState, get_chess_small_config

        config = get_chess_small_config()
        config.mcts.num_simulations = 50  # Fast for UI

        state = ChessGameState.from_fen(fen)
        agent = ChessEnsembleAgent(config)

        # Run async in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                agent.get_best_move(state, temperature=0.1, use_ensemble=False)
            )
            return response.best_move, {
                "value": response.value_estimate,
                "confidence": response.confidence,
                "routing": response.routing_decision.primary_agent.value,
                "reasoning": response.routing_decision.reasoning,
                "top_moves": dict(list(response.move_probabilities.items())[:5]),
            }
        finally:
            loop.close()

    except Exception as e:
        logger.warning(f"Ensemble agent failed, using fallback: {e}")

    # Fallback: simple heuristic move selection
    import random

    # Prefer captures and center moves
    scored_moves = []
    for move in legal_moves:
        score = random.random() * 0.1  # Base randomness

        # Prefer captures
        if board.is_capture(move):
            score += 0.5

        # Prefer center squares
        to_sq = move.to_square
        file, rank = to_sq % 8, to_sq // 8
        center_dist = abs(file - 3.5) + abs(rank - 3.5)
        score += (7 - center_dist) * 0.05

        # Prefer checks
        board.push(move)
        if board.is_check():
            score += 0.3
        board.pop()

        scored_moves.append((move, score))

    scored_moves.sort(key=lambda x: -x[1])
    best_move = scored_moves[0][0]

    return best_move.uci(), {
        "value": 0.0,
        "confidence": 0.5,
        "routing": "fallback",
        "reasoning": "Using simple heuristic (ensemble agent not available)",
        "top_moves": {m.uci(): s for m, s in scored_moves[:5]},
    }


def format_analysis(move: str, analysis: dict[str, Any]) -> str:
    """Format AI analysis for display."""
    lines = [
        f"AI plays: **{move}**",
        "",
        f"Evaluation: {analysis.get('value', 0):.2f}",
        f"Confidence: {analysis.get('confidence', 0):.1%}",
        f"Agent: {analysis.get('routing', 'unknown')}",
        "",
        f"Reasoning: {analysis.get('reasoning', 'N/A')}",
        "",
        "Top moves considered:",
    ]

    top_moves = analysis.get("top_moves", {})
    for m, score in list(top_moves.items())[:5]:
        lines.append(f"  {m}: {score:.1%}")

    return "\n".join(lines)


def undo_move() -> tuple[str, str, str, str]:
    """Undo the last two moves (player + AI).

    Returns:
        (board_html, status, move_history, analysis)
    """
    global _session

    if len(_session.move_history) < 2:
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            "Cannot undo: not enough moves",
        )

    try:
        import chess

        # Replay all moves except last 2
        board = chess.Board()
        moves_to_keep = _session.move_history[:-2]

        for move_uci in moves_to_keep:
            board.push(chess.Move.from_uci(move_uci))

        _session.fen = board.fen()
        _session.move_history = moves_to_keep
        _session.game_over = False
        _session.result = ""

        last_move = moves_to_keep[-1] if moves_to_keep else None

        return (
            render_board_html(_session.fen, last_move, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            "Undone last two moves",
        )

    except Exception as e:
        return (
            render_board_html(_session.fen, flipped=(_session.player_color == "black")),
            get_game_status(_session),
            format_move_history(_session.move_history),
            f"Error undoing: {str(e)}",
        )


def create_chess_ui() -> gr.Blocks:
    """Create the Gradio chess UI."""
    with gr.Blocks(
        title="Chess vs AlphaZero AI",
        theme=gr.themes.Soft(),
        css="""
        .chess-container { max-width: 1200px; margin: auto; }
        .move-input { font-family: monospace; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # Chess vs AlphaZero-Style AI

            Play chess against an AI using HRM, TRM, and Neural MCTS ensemble agents.

            **How to play:**
            1. Select your color and click "New Game"
            2. Enter moves in UCI format (e.g., `e2e4`, `g1f3`, `e7e8q` for promotion)
            3. Click "Make Move" or press Enter
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                board_display = gr.HTML(
                    value=render_board_html(_session.fen),
                    label="Chess Board",
                    elem_id="board-container",
                )

                with gr.Row():
                    move_input = gr.Textbox(
                        label="Your Move (UCI format)",
                        placeholder="e.g., e2e4",
                        elem_classes=["move-input"],
                        elem_id="move-input",
                    )
                    move_btn = gr.Button("Make Move", variant="primary", elem_id="move-btn")

                with gr.Row():
                    color_select = gr.Radio(
                        choices=["white", "black"],
                        value="white",
                        label="Play as",
                        elem_id="color-select",
                    )
                    new_game_btn = gr.Button("New Game", variant="secondary", elem_id="new-game-btn")
                    undo_btn = gr.Button("Undo", elem_id="undo-btn")

            with gr.Column(scale=1):
                status_display = gr.Textbox(
                    label="Game Status",
                    value="White to move",
                    interactive=False,
                    elem_id="status-display",
                )

                history_display = gr.Textbox(
                    label="Move History",
                    value="No moves yet",
                    interactive=False,
                    lines=10,
                    elem_id="history-display",
                )

                analysis_display = gr.Markdown(
                    value="*AI analysis will appear here*",
                    label="AI Analysis",
                    elem_id="analysis-display",
                )

        # Event handlers
        move_btn.click(
            fn=apply_player_move,
            inputs=[move_input],
            outputs=[board_display, status_display, history_display, analysis_display],
        ).then(
            fn=lambda: "",
            outputs=[move_input],
        )

        move_input.submit(
            fn=apply_player_move,
            inputs=[move_input],
            outputs=[board_display, status_display, history_display, analysis_display],
        ).then(
            fn=lambda: "",
            outputs=[move_input],
        )

        new_game_btn.click(
            fn=initialize_game,
            inputs=[color_select],
            outputs=[board_display, status_display, history_display, analysis_display],
        )

        undo_btn.click(
            fn=undo_move,
            outputs=[board_display, status_display, history_display, analysis_display],
        )

        # Initialize on load
        demo.load(
            fn=lambda: initialize_game("white"),
            outputs=[board_display, status_display, history_display, analysis_display],
        )

    return demo


def main() -> None:
    """Launch the chess UI."""
    import argparse

    parser = argparse.ArgumentParser(description="Chess Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    demo = create_chess_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
