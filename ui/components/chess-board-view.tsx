"use client";

import { useState, useEffect, useCallback } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SearchResult {
    best_move: string;
    root_value: number;
    simulations: number;
    nodes_explored: number;
    policy: Record<string, number>;
}

export default function ChessBoardView() {
    const [game, setGame] = useState(new Chess());
    const [fen, setFen] = useState(game.fen());
    const [aiThinking, setAiThinking] = useState(false);
    const [lastStats, setLastStats] = useState<SearchResult | null>(null);
    const [history, setHistory] = useState<{ fen: string, policy: Record<string, number> }[]>([]);

    const makeAMove = useCallback((move: string | { from: string; to: string; promotion?: string }) => {
        try {
            const gameCopy = new Chess(game.fen());
            const result = gameCopy.move(move);
            if (result) {
                setGame(gameCopy);
                setFen(gameCopy.fen());
                return result;
            }
        } catch {
            return null;
        }
        return null;
    }, [game]);

    const triggerAiMove = useCallback(async (currentFen: string) => {
        setAiThinking(true);
        try {
            const res = await fetch("http://localhost:8000/chess/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fen: currentFen, simulations: 400 }),
            });

            if (!res.ok) throw new Error("API Failed");

            const data: SearchResult = await res.json();
            setLastStats(data);

            // Record step for feedback
            setHistory(prev => [...prev, { fen: currentFen, policy: data.policy }]);

            makeAMove(data.best_move);

        } catch (err) {
            console.error(err);
        } finally {
            setAiThinking(false);
        }
    }, [makeAMove]);

    function onDrop(sourceSquare: string, targetSquare: string) {
        if (aiThinking) return false;

        const move = makeAMove({
            from: sourceSquare,
            to: targetSquare,
            promotion: "q",
        });

        if (move === null) return false;

        return true;
    }

    useEffect(() => {
        // Trigger AI if Black's turn
        if (game.turn() === 'b' && !game.isGameOver()) {
            triggerAiMove(game.fen());
        }

        // Check Game Over to send feedback
        if (game.isGameOver()) {
            const sendFeedback = async () => {
                if (history.length === 0) return;

                let outcome = 0.0;
                if (game.isCheckmate()) {
                    outcome = game.turn() === 'b' ? 1.0 : -1.0; // If Black to move and mated, White won
                }

                try {
                    await fetch("http://localhost:8000/chess/feedback", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            fen_history: history.map(h => h.fen),
                            policy_history: history.map(h => h.policy),
                            outcome: outcome
                        })
                    });
                    console.log("Game feedback sent!");
                } catch (e) {
                    console.error("Failed to send feedback", e);
                }
            };
            sendFeedback();
        }
    }, [game, fen, triggerAiMove, history]);

    return (
        <div className="flex flex-col md:flex-row gap-8 w-full max-w-4xl">
            <div className="flex-1 aspect-square max-w-[500px]">
                <Chessboard
                    // @ts-expect-error type definition mismatch
                    position={fen}
                    onPieceDrop={onDrop}
                    arePiecesDraggable={!aiThinking && game.turn() === 'w'}
                />
            </div>

            <div className="flex-1 flex flex-col gap-4">
                <div className="p-4 border rounded-lg bg-white shadow-sm">
                    <h3 className="font-bold text-lg mb-2">Game Status</h3>
                    <div className="space-y-2">
                        <p>Turn: <span className={cn("font-mono font-bold", game.turn() === 'w' ? "text-blue-600" : "text-red-600")}>
                            {game.turn() === 'w' ? "White (Human)" : "Black (AI)"}
                        </span></p>
                        {aiThinking && <p className="animate-pulse text-yellow-600">Thinking...</p>}
                        {game.isGameOver() && <p className="text-red-500 font-bold">Game Over!</p>}
                    </div>
                </div>

                {lastStats && (
                    <div className="p-4 border rounded-lg bg-slate-50 shadow-sm">
                        <h3 className="font-bold text-lg mb-2">Neural MCTS Stats</h3>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                            <div className="text-gray-500">Best Move:</div>
                            <div className="font-mono font-bold">{lastStats.best_move}</div>

                            <div className="text-gray-500">Value (Win%):</div>
                            <div className="font-mono">{lastStats.root_value.toFixed(3)}</div>

                            <div className="text-gray-500">Simulations:</div>
                            <div className="font-mono">{lastStats.simulations}</div>

                            <div className="text-gray-500">Nodes:</div>
                            <div className="font-mono">{lastStats.nodes_explored}</div>
                        </div>
                    </div>
                )}

                <Button
                    variant="outline"
                    onClick={() => {
                        const newGame = new Chess();
                        setGame(newGame);
                        setFen(newGame.fen());
                        setLastStats(null);
                        setHistory([]);
                    }}
                >
                    Reset Game
                </Button>
            </div>
        </div>
    );
}
