# skills/ Format Guide

This folder contains **skills** - contextual knowledge that auto-loads based on conversation triggers.

## Folder Structure

```
skills/
├── FORMAT.md                    # This guide
├── skill-name/                  # Each skill is a folder
│   ├── SKILL.md                 # Main skill file (required)
│   └── references/              # Optional: detailed reference docs
│       ├── topic1.md
│       ├── topic2.md
│       └── examples.md
```

## SKILL.md Format

### 1. YAML Frontmatter (Required)

```yaml
---
name: skill-name
description: Detailed description explaining when this skill should be loaded. Include keywords and triggers. This is used for semantic matching to auto-load the skill.
references:                      # Optional: sub-skills to also load
  - reference1
  - reference2
---
```

**Fields:**
- `name` (Required): Skill identifier (matches folder name)
- `description` (Required): **Critical** - This is how the AI decides when to load the skill. Include:
  - What the skill covers
  - Keywords that should trigger loading
  - Use cases and scenarios
- `references` (Optional): List of reference folders to auto-load with this skill

### 2. Markdown Body

```markdown
# Skill Title

Brief overview of what this skill provides.

## FIRST: Verify Prerequisites

If the skill requires specific tools, setup, or dependencies, check them first.

## Key Concepts

Core concepts the AI needs to understand.

## Quick Reference

| Task | How to Do It |
|------|--------------|
| Task 1 | Method/API |
| Task 2 | Method/API |

## Common Patterns

Code examples and patterns.

## Detailed References

Links to reference files for deeper information:
- **[references/topic1.md](references/topic1.md)** - Description
- **[references/topic2.md](references/topic2.md)** - Description

## Best Practices

Numbered list of recommendations.
```

## references/ Subfolder

For complex skills, break detailed information into separate files:

```
references/
├── examples.md          # Code examples and templates
├── patterns.md          # Common patterns and architectures  
├── troubleshooting.md   # Common issues and solutions
├── api.md               # API reference
└── configuration.md     # Configuration options
```

Each reference file is plain Markdown without special frontmatter.

## Example SKILL.md

```markdown
---
name: data-analysis
description: Analyze scientific datasets using Python. Load when working with pandas, numpy, matplotlib, statistical analysis, data visualization, CSV/Parquet files, or exploratory data analysis (EDA).
---

# Data Analysis Skill

Guidance for analyzing scientific datasets with Python.

## FIRST: Verify Environment

```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Quick Reference

| Task | Code |
|------|------|
| Load CSV | `pd.read_csv('file.csv')` |
| Summary stats | `df.describe()` |
| Plot histogram | `df['column'].hist()` |

## Common Patterns

### Exploratory Data Analysis

\`\`\`python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.info())
print(df.describe())
df.hist(figsize=(12, 8))
plt.tight_layout()
\`\`\`

## Detailed References

- **[references/visualization.md](references/visualization.md)** - Plotting and charts
- **[references/statistics.md](references/statistics.md)** - Statistical tests

## Best Practices

1. Always check for missing values first
2. Use appropriate data types
3. Document your analysis steps
```

## How Skills Auto-Load

Skills load automatically when the conversation matches their description. Write descriptions that include:

1. **Specific technologies**: "pandas", "numpy", "matplotlib"
2. **Task types**: "data analysis", "visualization", "statistical testing"
3. **File types**: "CSV", "Parquet", "HDF5"
4. **Domains**: "scientific computing", "machine learning"

The more specific your description, the better the skill matches relevant conversations.

## Tips

1. **One skill = one domain** - Don't mix unrelated topics
2. **Rich descriptions** - More keywords = better matching
3. **Code examples** - Show, don't just tell
4. **Link references** - Keep SKILL.md focused, details in references/
5. **Prerequisites first** - Check requirements before diving in
