import { Button } from "@/components/ui/button";
import MCTSTreeView from "@/components/mcts-tree-view";
import ChessBoardView from "@/components/chess-board-view";

export default function Home() {
    return (
        <main className="flex min-h-screen flex-col items-center justify-between p-24 bg-slate-50">
            <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
                <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
                    LangGraph MCTS Orchestrator
                </p>
                <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
                    <span className="flex place-items-center gap-2 p-8 lg:p-0">
                        Status: <span className="text-green-500 font-bold">Online</span>
                    </span>
                </div>
            </div>

            <div className="w-full max-w-5xl mt-10">
                <h2 className="text-2xl font-bold mb-4">Neural Search</h2>
                <div className="flex gap-4 mb-8">
                    <input
                        type="text"
                        placeholder="Enter programming problem..."
                        className="flex-1 p-2 border rounded-md"
                    />
                    <Button>Start Search</Button>
                </div>

                <h3 className="text-xl font-semibold mb-2">Real-time MCTS Tree</h3>
                <MCTSTreeView />

                <div className="my-12 border-t pt-8">
                    <h2 className="text-2xl font-bold mb-4">Chess AlphaZero</h2>
                    <ChessBoardView />
                </div>
            </div>

            <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-4 lg:text-left">
                {/* Future Stats widgets */}
            </div>
        </main>
    );
}
