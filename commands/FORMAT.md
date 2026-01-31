# commands/ Format Guide

This folder contains **slash commands** - user-invocable actions that users explicitly call via `/command-name`.

## File Naming

- Files must be `.md` (Markdown)
- Filename becomes the command name: `build-agent.md` → `/build-agent`
- Use kebab-case: `analyze-data.md`, `run-experiment.md`

## File Structure

Each command file has two parts:

### 1. YAML Frontmatter (Required)

```yaml
---
description: Brief description shown in command autocomplete
argument-hint: [optional-argument-placeholder]
allowed-tools: [Read, Glob, Grep, Bash, Write, Edit, WebFetch]
---
```

**Fields:**
- `description` (Required): One-line description of what the command does
- `argument-hint` (Optional): Placeholder text for expected argument
- `allowed-tools` (Optional): List of tools the command can use

### 2. Markdown Body

```markdown
# Command Title

Brief explanation of what this command does.

## Arguments

The user invoked this command with: $ARGUMENTS

## Instructions

Step-by-step guide for the AI when this command is invoked:

1. First, do X
2. Then, do Y
3. Finally, do Z

## Capabilities

- Capability 1
- Capability 2

## Example Usage

\`\`\`
/command-name argument1
/command-name different argument
\`\`\`
```

## Example Command

```markdown
---
description: Analyze a scientific dataset and generate a report
argument-hint: [path-to-dataset]
allowed-tools: [Read, Glob, Grep, Bash, Write]
---

# Analyze Scientific Data

This command analyzes a dataset and generates a summary report.

## Arguments

The user invoked this command with: $ARGUMENTS

## Instructions

1. Read the skill file at `data-analysis/SKILL.md` for methodology
2. Load the dataset from the provided path
3. Generate statistical summary
4. Create visualization recommendations
5. Output a markdown report

## Example Usage

\`\`\`
/analyze-data ./experiments/results.csv
/analyze-data ./data/measurements.parquet
\`\`\`
```

## Best Practices

1. **Keep descriptions concise** - They appear in autocomplete
2. **Use $ARGUMENTS** - This gets replaced with user's input
3. **Reference skills** - Point to relevant SKILL.md files for detailed guidance
4. **Provide examples** - Show 2-3 typical usage patterns
5. **Limit scope** - One command = one focused action
