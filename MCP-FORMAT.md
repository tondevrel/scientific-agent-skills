# .mcp.json Format Guide

This file configures MCP (Model Context Protocol) servers that provide additional tools and capabilities to the AI agent.

## File Location

Place `.mcp.json` in the repository root.

## Format

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["mcp-package-name", "arg1", "arg2"]
    },
    "another-server": {
      "command": "npx", 
      "args": ["mcp-remote", "https://remote.mcp.server.com/mcp"]
    }
  }
}
```

## Server Types

### Local MCP Server (npm package)

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "npx",
      "args": ["-y", "chrome-devtools-mcp@latest"]
    }
  }
}
```

### Remote MCP Server

```json
{
  "mcpServers": {
    "docs-server": {
      "command": "npx",
      "args": ["mcp-remote", "https://docs.example.com/mcp"]
    }
  }
}
```

### Custom Command

```json
{
  "mcpServers": {
    "custom-server": {
      "command": "python",
      "args": ["-m", "my_mcp_server", "--port", "3000"]
    }
  }
}
```

## Common MCP Servers

| Server | Purpose | Package/URL |
|--------|---------|-------------|
| Chrome DevTools | Browser automation, performance | `chrome-devtools-mcp@latest` |
| Filesystem | File operations | `@anthropic/mcp-server-filesystem` |
| GitHub | Repository operations | `@anthropic/mcp-server-github` |
| PostgreSQL | Database queries | `@anthropic/mcp-server-postgres` |

## When to Use

- **Use MCP servers** when you need tools beyond basic file/terminal operations
- **Remote servers** for accessing external APIs or documentation
- **Local servers** for local tool integration (browsers, databases, etc.)

## Notes

- MCP servers must be installed before use
- Remote servers require network access
- Remove `.mcp.json` if you don't need MCP servers
