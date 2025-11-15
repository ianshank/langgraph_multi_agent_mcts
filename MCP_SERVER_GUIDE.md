# MCP Server Guide

## Overview

The MCP (Model Context Protocol) Server provides a standardized interface to the Multi-Agent MCTS Framework. It exposes tools for running MCTS searches, querying individual agents, and managing artifacts through a JSON-RPC protocol over stdio.

## LM Studio Integration

The MCP server is fully integrated with LM Studio for local model inference:

- **Model**: llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b
- **Endpoint**: http://127.0.0.1:1234/v1
- **Provider**: lmstudio

### Prerequisites

1. Install and run LM Studio
2. Load the model in LM Studio
3. Ensure the API server is enabled (default port: 1234)

## Configuration

The server is configured via `mcp_config.json`:

```json
{
  "mcpServers": {
    "mcts-framework": {
      "command": "python",
      "args": ["tools/mcp/server.py"],
      "cwd": "C:\\Users\\iansh\\OneDrive\\Documents\\GitHub\\langgraph_multi_agent_mcts",
      "env": {
        "LLM_PROVIDER": "lmstudio",
        "LMSTUDIO_BASE_URL": "http://localhost:1234/v1",
        "LMSTUDIO_MODEL": "llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b",
        "MCTS_ITERATIONS": "100",
        "MCTS_C": "1.414",
        "SEED": "42",
        "LOG_LEVEL": "INFO",
        "HTTP_TIMEOUT_SECONDS": "120",
        "HTTP_MAX_RETRIES": "3"
      }
    }
  }
}
```

## Starting the Server

To start the MCP server:

```bash
# Windows PowerShell
cd "C:\Users\iansh\OneDrive\Documents\GitHub\langgraph_multi_agent_mcts"
$env:LLM_PROVIDER = "lmstudio"
$env:LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
# ... set other environment variables ...
python tools/mcp/server.py

# Or use the configuration directly
python tools/mcp/server.py
```

The server will output an initialization message and then listen for JSON-RPC requests on stdin.

## Available Tools

### 1. run_mcts
Execute MCTS search with the multi-agent framework.

**Parameters:**
- `query` (string, required): The query to process
- `iterations` (int, default: 100): Number of MCTS iterations (1-10000)
- `exploration_weight` (float, default: 1.414): UCB1 exploration constant (0.0-10.0)
- `seed` (int, optional): Random seed for determinism
- `use_rag` (bool, default: false): Enable RAG context retrieval

**Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "run_mcts",
    "arguments": {
      "query": "How to implement a caching system?",
      "iterations": 50,
      "exploration_weight": 1.414,
      "seed": 42
    }
  }
}
```

### 2. query_agent
Query a specific agent (HRM, TRM, or MCTS).

**Parameters:**
- `agent_type` (string, required): Agent type: "hrm", "trm", or "mcts"
- `query` (string, required): Query to send to the agent
- `temperature` (float, default: 0.7): Sampling temperature (0.0-2.0)
- `max_tokens` (int, optional): Maximum tokens in response

**Example:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "query_agent",
    "arguments": {
      "agent_type": "hrm",
      "query": "Break down the task of building a REST API",
      "temperature": 0.7
    }
  }
}
```

### 3. get_artifact
Retrieve a stored artifact by ID.

**Parameters:**
- `artifact_id` (string, required): ID of the artifact to retrieve
- `artifact_type` (string, default: "mcts_stats"): Type: "mcts_stats", "config", "trace", or "log"

### 4. list_artifacts
List available artifacts.

**Parameters:**
- `artifact_type` (string, optional): Filter by type
- `limit` (int, default: 50): Maximum number of results (1-1000)

### 5. get_config
Get current framework configuration.

**Parameters:** None

### 6. health_check
Check health of framework and LLM provider.

**Parameters:** None

## Response Format

Successful responses:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "success": true,
    "data": "..."
  }
}
```

Error responses:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Error description"
  }
}
```

## Integration Example

### Python Client
```python
import json
import subprocess

def call_mcp_tool(tool_name, arguments):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    # Send to MCP server via subprocess
    # ... implementation ...
```

### Node.js Client
```javascript
const { spawn } = require('child_process');

const mcp = spawn('python', ['tools/mcp/server.py']);

function callTool(toolName, args) {
    const request = {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
            name: toolName,
            arguments: args
        }
    };
    
    mcp.stdin.write(JSON.stringify(request) + '\n');
}
```

## Troubleshooting

1. **LLM Provider Not Available**
   - Ensure LMStudio is running on http://localhost:1234
   - Check the model is loaded in LMStudio

2. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes the project root

3. **JSON Parse Errors**
   - Ensure each request is a single line of valid JSON
   - Check for proper escaping of special characters

## Architecture

The MCP server consists of:
- `tools/mcp/server.py`: Main server implementation
- `src/adapters/llm.py`: LLM client adapters (LMStudio, OpenAI)
- `src/config/settings.py`: Configuration management
- `src/framework/mcts/`: MCTS implementation

All tools use async functions with Pydantic validation for type safety and input sanitization.

## Security Considerations

- Input sanitization is performed on all user queries
- File paths are validated to prevent directory traversal
- API keys and sensitive data should be set via environment variables
- Consider running the server in a sandboxed environment for production use
