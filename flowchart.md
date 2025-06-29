# AirPilot Blackbox Flowchart

## Overview

**AirPilot Blackbox** is a comprehensive Claude AI usage analysis and cost tracking tool that processes Claude conversation logs and provides detailed analytics with cost calculations.

### Key Functionality

- Analyzes Claude usage logs from `~/.claude` or Docker containers
- Calculates accurate costs using real-time or hardcoded pricing
- Provides advanced analytics (cache efficiency, response times, tool usage)
- Supports session deduplication across multiple sources
- Exports results to JSON with rich terminal UI

## Flow Description

These flowcharts illustrate the complete data processing pipeline of AirPilot Blackbox, from input sources through parsing, analysis, cost calculation, and output generation.

## FlowChart 10,000 foot

```mermaid
flowchart LR
   A[Claude Logs] --> B[Calculate Costs]
        B --> C[Show Reports]
```

**Bottom Line**: Turn AI spending visibility into cost savings.

## Flowchart Summary

```mermaid
flowchart TB
    A[AirPilot Blackbox<br/>Usage Analysis Tool] --> B[Data Collection]

    B --> C[Local Claude Data<br/>~/.claude directory]
    B --> D[Docker Containers<br/>DevContainer environments]

    C --> E[Data Processing<br/>& Analysis]
    D --> E

    E --> F[Cost Calculation<br/>Token usage to USD]
    F --> G[Advanced Analytics<br/>Performance & Efficiency]

    G --> H[Reporting & Insights]

    H --> I[Terminal Dashboard<br/>Rich visual tables]
    H --> J[JSON Export<br/>Structured data output]
    H --> K[Executive Summary<br/>High-level metrics]

```

## Flowchart System Architecture

```mermaid
flowchart TD
    A[User Input<br/>CLI Command] --> B{Data Source?}

    B -->|--claude-dir| C[Local Directory<br/>~/.claude/projects/]
    B -->|--docker| D[Docker Containers<br/>Search & Copy]

    C --> E[Find JSONL Files<br/>*.jsonl]
    D --> F[Docker Process<br/>Copy to Temp Dir]
    F --> E

    E --> G[Global Parser<br/>parse_all_files_globally]

    G --> H[Parse JSONL Lines<br/>JSON.loads each line]
    H --> I{Entry Type?}

    I -->|user| J[Process User Message<br/>Track CWD, Timestamps]
    I -->|assistant| K[Process Assistant Message<br/>Extract Token Usage]

    J --> L[Update Session Stats]
    K --> M[Deduplication Check<br/>messageId:requestId]

    M -->|Duplicate| N[Skip Entry]
    M -->|New| O[Update Statistics]

    O --> P[Update Counters<br/>- Total tokens<br/>- By model<br/>- By session<br/>- By date<br/>- Tool usage]

    L --> P
    P --> Q[Statistics Complete]

    Q --> R[Cost Calculator<br/>Calculate costs]
    R --> S{Pricing Source?}

    S -->|LiteLLM| T[Fetch Real-time Pricing<br/>pricing_fetcher.py]
    S -->|Hardcoded| U[Use Built-in Pricing<br/>MODEL_PRICING dict]

    T --> V[Normalize Model Names<br/>opus-4, sonnet-4, etc.]
    U --> V

    V --> W[Calculate Model Costs<br/>- Input tokens × rate<br/>- Output tokens × rate<br/>- Cache tokens × rate]

    W --> X[Calculate Session Costs<br/>Weighted by models used]
    X --> Y[Calculate Daily Costs<br/>Aggregate by date]

    Y --> Z[Analytics Processing<br/>analytics.py]
    Z --> AA[Cache Analytics<br/>- Hit rate<br/>- Efficiency<br/>- ROI calculation]
    Z --> BB[Response Time Analysis<br/>- User→Assistant pairs<br/>- Model performance]
    Z --> CC[Tool Usage Analytics<br/>- Usage patterns<br/>- Combinations<br/>- Cost per tool]

    AA --> DD[Display Results<br/>Rich Terminal UI]
    BB --> DD
    CC --> DD

    DD --> EE{Output Options?}

    EE -->|--output| FF[Export to JSON<br/>Serializable format]
    EE -->|--summary-only| GG[Show Summary Only<br/>Overall + Model breakdown]
    EE -->|--tools| HH[Show Tool Usage<br/>Enhanced tool analytics]
    EE -->|--cache| II[Show Cache Analytics<br/>Efficiency metrics]
    EE -->|--response-times| JJ[Show Response Times<br/>Performance analysis]
    EE -->|--full| KK[Show All Analytics<br/>Complete analysis]
    EE -->|Default| LL[Standard Display<br/>Overview + breakdowns]

    FF --> MM[Complete]
    GG --> MM
    HH --> MM
    II --> MM
    JJ --> MM
    KK --> MM
    LL --> MM

    N --> Q

    style A fill:#e1f5fe
    style MM fill:#c8e6c9
    style G fill:#fff3e0
    style R fill:#fff3e0
    style Z fill:#fff3e0
    style DD fill:#f3e5f5
    style M fill:#ffebee
```



### Key Business Benefits

- **Cost Transparency**: Real-time cost tracking and forecasting
- **Usage Insights**: Detailed analytics on AI tool adoption and efficiency
- **Performance Monitoring**: Response time and cache efficiency metrics
- **Multi-Environment Support**: Works across local and containerized development
- **ROI Analysis**: Quantify AI investment returns and optimization opportunities


### Key Processing Stages

1. **Input Processing**: Handles both local `~/.claude` directories and Docker container sources
2. **File Discovery**: Locates all JSONL files containing Claude conversation logs
3. **Parsing & Deduplication**: Processes JSON entries with global message deduplication
4. **Statistics Collection**: Aggregates usage data by model, session, date, and tools
5. **Cost Calculation**: Applies real-time or hardcoded pricing to usage statistics
6. **Analytics Processing**: Generates advanced metrics for cache, response times, and tools
7. **Output Generation**: Displays results via rich terminal UI or exports to JSON
