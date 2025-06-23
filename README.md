# AirPilot Blackbox

Analyze your Claude AI usage logs and calculate costs from `~/.claude` or Docker containers.

## Quick Start

### Option 1: Run directly with uvx (recommended)

```sh
uvx airpilot-blackbox
# or use the short alias
blackbox
```

### Option 2: Clone and run

```sh
git clone https://github.com/shaneholloman/airpilot-blackbox.git
cd airpilot-blackbox
uv run .
```

## Runtime Usage

```sh
# Analyze local ~/.claude directory (default)
blackbox
# or
uvx airpilot-blackbox

# Analyze Docker containers
blackbox --docker

# Export results to JSON
blackbox --output results.json

# Show only summary
blackbox --summary-only

# Show tool usage statistics
blackbox --tools

# Show cache efficiency analytics
blackbox --cache

# Show response time analysis
blackbox --response-times

# Show all enhanced analytics
blackbox --full

# Limit items shown in tables
blackbox --limit 5
```

## uvx usage

```sh
# Analyze local ~/.claude directory (default)
uvx airpilot-blackbox

# Analyze Docker containers
uvx airpilot-blackbox --docker

# Export results to JSON
uvx airpilot-blackbox --output results.json

# Show only summary
uvx airpilot-blackbox --summary-only

# Show tool usage statistics
uvx airpilot-blackbox --tools

# Show cache efficiency analytics
uvx airpilot-blackbox --cache

# Show response time analysis
uvx airpilot-blackbox --response-times

# Show all enhanced analytics
uvx airpilot-blackbox --full

# Limit items shown in tables
uvx airpilot-blackbox --limit 5
```

## Features

- **Zero Installation**: Just clone and run with `uv`
- **Comprehensive Stats**: Token usage, costs, sessions, daily trends
- **Docker Support**: Analyze usage from devcontainers (auto-analyzes all containers)
- **Session Deduplication**: Automatically detects and merges duplicate sessions across sources
- **Rich Terminal UI**: Beautiful tables and formatting
- **Cost Tracking**: Automatic calculation based on current Claude API pricing
- **Cache Analytics**: Track cache efficiency, ROI, and savings
- **Response Time Analysis**: Monitor performance by model and percentiles
- **Enhanced Tool Analytics**: Cost per tool, usage patterns, and combinations

## Example Output

```sh
uv run blackbox
╭────────────────────────╮
│ AirPilot Blackbox      │
│ Analyse AirPilot usage │
╰────────────────────────╯

Found 232 JSONL files across 1 source(s)
  Parsing usage logs...


Overall Usage Statistics
 Total Messages                20,132
 Input Tokens                 155,998
 Output Tokens              1,834,052
 Cache Creation Tokens     73,508,335
 Cache Read Tokens      1,532,465,059
 Total Tokens           1,607,963,444

 Total Cost                   $982.98

Model Usage Breakdown
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Model                    ┃ Messages ┃   Input ┃    Output ┃         Cache ┃    Cost ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ claude-sonnet-4-20250514 │   18,298 │ 151,274 │ 1,763,274 │ 1,510,144,603 │ $708.47 │
│ claude-opus-4-20250514   │    1,194 │   4,724 │    70,778 │    95,828,791 │ $274.51 │
│ <synthetic>              │      640 │       0 │         0 │             0 │ $0.0000 │
└──────────────────────────┴──────────┴─────────┴───────────┴───────────────┴─────────┘

Daily Usage
┏━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Date       ┃ Messages ┃ Sessions ┃      Tokens ┃    Cost ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ 2025-06-23 │      335 │        3 │  23,717,050 │  $51.39 │
│ 2025-06-22 │    1,256 │       30 │  78,414,970 │ $202.86 │
│ 2025-06-21 │    1,014 │        2 │  90,414,152 │ $217.63 │
│ 2025-06-20 │      380 │        4 │  21,862,784 │  $69.93 │
│ 2025-06-19 │    1,048 │        7 │  79,102,093 │ $233.74 │
│ 2025-06-18 │       30 │        1 │   1,663,801 │   $6.13 │
│ 2025-06-17 │      153 │        5 │   6,541,711 │  $24.51 │
│ 2025-06-16 │    1,774 │       19 │ 130,643,490 │ $275.64 │
│ 2025-06-15 │      315 │        2 │  26,584,711 │  $66.19 │
│ 2025-06-14 │    2,440 │       25 │ 190,579,840 │ $411.25 │
└────────────┴──────────┴──────────┴─────────────┴─────────┘

Session Breakdown
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Session ID  ┃ Messages ┃ Duration ┃ Models              ┃   Cost ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 76966e72... │      422 │    3h 9m │ opus, sonnet, +1    │ $88.22 │
│ 81eab4b0... │      464 │   5h 20m │ opus, <synthetic>   │ $87.72 │
│ 126eb98b... │      320 │   3h 21m │ opus, <synthetic>   │ $71.68 │
│ 6ddd6588... │      176 │   4h 55m │ opus, <synthetic>   │ $57.34 │
│ 50a9c408... │      962 │  22h 41m │ sonnet, <synthetic> │ $42.44 │
│ 63dc55b9... │      838 │   5h 48m │ sonnet, <synthetic> │ $39.39 │
│ 71c64fb4... │    1,169 │   3h 53m │ sonnet              │ $38.31 │
│ 12606cef... │    1,069 │   7h 15m │ sonnet, <synthetic> │ $37.30 │
│ e42f0435... │      918 │    8h 7m │ sonnet, <synthetic> │ $35.40 │
│ 7a39bcd8... │      791 │   6h 52m │ sonnet, <synthetic> │ $31.71 │
└─────────────┴──────────┴──────────┴─────────────────────┴────────┘
```

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (optional, for container support)

## License

MIT
