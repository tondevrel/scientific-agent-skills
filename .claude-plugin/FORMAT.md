# .claude-plugin/ Format Guide

This folder contains configuration files for Claude Code plugin integration.

## Files

### plugin.json (Required)
Plugin metadata for Claude Code.

```json
{
  "name": "your-plugin-name",
  "description": "Brief description of what skills this plugin provides",
  "version": "1.0.0",
  "author": {
    "name": "Author or Organization Name"
  }
}
```

**Fields:**
- `name`: Plugin identifier (lowercase, hyphens allowed)
- `description`: One-line summary of capabilities
- `version`: Semantic version (e.g., "1.0.0")
- `author.name`: Creator name

### marketplace.json (Optional)
Required only if publishing to Claude Code marketplace.

```json
{
  "$schema": "https://code.claude.com/schemas/marketplace.json",
  "name": "your-plugin-name",
  "owner": {
    "name": "Organization Name",
    "url": "https://your-website.com"
  },
  "plugins": [
    {
      "name": "your-plugin-name",
      "source": "./",
      "description": "Brief description"
    }
  ]
}
```

## Notes

- Keep `name` consistent across plugin.json and marketplace.json
- The `source` field typically points to "./" (repository root)
- Update version when making significant changes
