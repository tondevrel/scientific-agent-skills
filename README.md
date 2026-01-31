# Scientific Agent Skills

A collection of Agent Skills for scientific computing, research workflows, and data analysis.

## Structure

```
scientific-agent-skills/
├── .claude-plugin/          # Claude Code plugin configuration
│   ├── plugin.json          # Plugin metadata
│   ├── marketplace.json     # Marketplace listing (optional)
│   └── FORMAT.md            # Format documentation
├── commands/                # Slash commands (/command-name)
│   ├── FORMAT.md            # Format documentation
│   └── *.md                 # Command files
├── skills/                  # Auto-loading contextual skills
│   ├── FORMAT.md            # Format documentation
│   └── skill-name/          # Each skill is a folder
│       ├── SKILL.md         # Main skill definition
│       └── references/      # Detailed reference docs
├── .mcp.json                # MCP server configuration (optional)
├── MCP-FORMAT.md            # MCP format documentation
└── README.md                # This file
```

## Installing

These skills work with any agent that supports the Agent Skills standard.

### Claude Code

```bash
/plugin marketplace add your-username/scientific-agent-skills
```

### npx skills

```bash
npx skills add https://github.com/your-username/scientific-agent-skills
```

### Manual Installation

Clone and copy to your agent's skill directory:

| Agent | Skill Directory |
|-------|-----------------|
| Claude Code | `~/.claude/skills/` |
| OpenCode | `~/.config/opencode/skill/` |
| OpenAI Codex | `~/.codex/skills/` |
| Pi | `~/.pi/agent/skills/` |

## Commands

Commands are user-invocable slash commands:

| Command | Description |
|---------|-------------|
| `/example-command` | Example command - replace with your own |

## Skills

Skills auto-load based on conversation context:

| Skill | Triggers |
|-------|----------|
| `example-skill` | Example triggers - replace with your own |

## Adding New Skills

1. Create a folder in `skills/` with your skill name
2. Add `SKILL.md` with frontmatter (name, description) and content
3. Optionally add `references/` folder for detailed docs
4. See `skills/FORMAT.md` for detailed instructions

## Adding New Commands

1. Create a `.md` file in `commands/`
2. Add frontmatter (description, argument-hint, allowed-tools)
3. Add markdown body with instructions
4. See `commands/FORMAT.md` for detailed instructions

## Format Guides

Each folder contains a `FORMAT.md` explaining the expected file format:

- [`.claude-plugin/FORMAT.md`](.claude-plugin/FORMAT.md) - Plugin configuration
- [`commands/FORMAT.md`](commands/FORMAT.md) - Slash commands
- [`skills/FORMAT.md`](skills/FORMAT.md) - Contextual skills
- [`MCP-FORMAT.md`](MCP-FORMAT.md) - MCP server configuration

## License

MIT
