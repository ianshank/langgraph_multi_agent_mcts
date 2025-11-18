# Configuration Files

This directory contains configuration files for the LangGraph Multi-Agent MCTS Framework.

## Files

- **mcp_config.json** - Active MCP (Model Context Protocol) server configuration
- **mcp_config.example.json** - Example MCP configuration template
- **mcp_config_template.json** - Template for generating new configurations

## Usage

Copy the example configuration and customize for your environment:

```bash
cp config/mcp_config.example.json config/mcp_config.json
# Edit mcp_config.json with your settings
```

## Environment Variables

For sensitive configuration like API keys, use environment variables instead of config files:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_PROVIDER=lmstudio
```

See `docs/API_CONFIGURATION_GUIDE.md` for detailed configuration options.
