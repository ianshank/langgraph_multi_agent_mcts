"use client";

import React, { useCallback } from 'react';
import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    Edge,
    Node,
} from 'reactflow';
import 'reactflow/dist/style.css';

interface MCTSTreeViewProps {
    initialNodes?: Node[];
    initialEdges?: Edge[];
}

const initialNodes: Node[] = [
    { id: '1', position: { x: 250, y: 0 }, data: { label: 'Root (Start)' } },
    { id: '2', position: { x: 100, y: 100 }, data: { label: 'Node A (Value: 0.8)' } },
    { id: '3', position: { x: 400, y: 100 }, data: { label: 'Node B (Value: 0.4)' } },
];

const initialEdges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2', animated: true },
    { id: 'e1-3', source: '1', target: '3' },
];

export default function MCTSTreeView({ initialNodes: propNodes, initialEdges: propEdges }: MCTSTreeViewProps) {
    const [nodes, , onNodesChange] = useNodesState(propNodes || initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(propEdges || initialEdges);

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        [setEdges],
    );

    return (
        <div style={{ width: '100%', height: '500px' }} className="border rounded-md bg-white">
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                fitView
            >
                <Controls />
                <MiniMap />
                <Background gap={12} size={1} />
            </ReactFlow>
        </div>
    );
}
