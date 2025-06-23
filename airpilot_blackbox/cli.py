"""CLI interface for AirPilot Blackbox."""

import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from dateutil import parser as date_parser
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from importlib.metadata import version as get_version
except ImportError:
    # Python < 3.8
    try:
        from importlib_metadata import version as get_version  # type: ignore
    except ImportError:

        def get_version(package: str) -> str:  # type: ignore
            return "unknown"


from .analytics import (
    analyze_tool_usage,
    calculate_cache_metrics,
    calculate_response_times,
)
from .parser import UsageParser
from .pricing import CostCalculator

console = Console()


class RichHelpFormatter:
    """Custom help formatter using Rich panels."""

    @staticmethod
    def format_help(ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format help text in a Rich panel."""
        # Get the original help text
        help_text = ctx.get_help()

        # Create a rich panel with the help content
        panel = Panel(
            help_text,
            title="[bold cyan]AirPilot BlackBox Help[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

        console.print(panel)
        ctx.exit()


class RichContext(click.Context):
    """Custom Click context that uses Rich for help formatting."""

    def get_help(self) -> str:
        """Override to customize help display."""
        if self.info_name is None:
            self.info_name = "blackbox"

        # Build help content manually for better formatting
        help_content = []

        # Usage line
        help_content.append(f"[bold]Usage:[/bold] {self.info_name} [OPTIONS]")
        help_content.append("")

        # Description
        if self.command.help:
            help_content.append("[bold]Description:[/bold]")
            help_content.append(f"  {self.command.help}")
            help_content.append("")

        # Options
        help_content.append("[bold]Options:[/bold]")

        # Get all options from the command
        for param in self.command.params:
            if isinstance(param, click.Option):
                # Format option names
                names = []
                if param.secondary_opts:
                    names.extend(param.secondary_opts)
                if param.opts:
                    names.extend(param.opts)

                name_str = ", ".join(names)

                # Add type info if needed
                if param.type.name != "flag" and param.type.name != "bool":
                    if param.type.name == "path":
                        name_str += " PATH"
                    elif param.type.name == "int":
                        name_str += " INTEGER"
                    else:
                        name_str += " TEXT"

                # Format help text
                help_line = f"  [cyan]{name_str:<20}[/cyan] {param.help or ''}"
                if param.default and param.type.name != "flag":
                    if (
                        param.default != "~/.claude" and param.default != 10
                    ):  # Don't show these defaults as they're in help text
                        help_line += f" [dim](default: {param.default})[/dim]"

                help_content.append(help_line)

        return "\n".join(help_content)


def parse_all_files_globally(all_files: List[Tuple[Path, str]]) -> Dict[str, Any]:
    """Parse all JSONL files with global deduplication."""
    stats: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_messages": 0,
        "by_model": defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "messages": 0,
                "requests": set(),
            }
        ),
        "by_session": defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "messages": 0,
                "user_messages": 0,
                "start_time": None,
                "end_time": None,
                "cwd": None,
                "models_used": set(),
            }
        ),
        "by_date": defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "messages": 0,
                "sessions": set(),
            }
        ),
        "errors": [],
        "tool_usage": defaultdict(int),
        "hourly_distribution": defaultdict(int),
        "all_messages": [],
        "seen_messages": set(),  # Global deduplication
    }

    # Parse each file
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing usage logs...", total=len(all_files))

        for file_path, source in all_files:
            try:
                # Extract session ID from file path
                session_id = file_path.stem

                with open(file_path, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                process_entry_global(data, stats, session_id, source)
                            except (json.JSONDecodeError, KeyError) as e:
                                stats["errors"].append(
                                    {
                                        "file": str(file_path),
                                        "line": line_num,
                                        "error": str(e),
                                        "source": source,
                                    }
                                )
            except Exception as e:
                stats["errors"].append(
                    {
                        "file": str(file_path),
                        "error": f"Failed to read file: {e}",
                        "source": source,
                    }
                )

            progress.update(task, advance=1)

    # Convert defaultdicts to regular dicts
    stats["by_model"] = dict(stats["by_model"])
    stats["by_session"] = dict(stats["by_session"])
    stats["by_date"] = dict(stats["by_date"])
    stats["tool_usage"] = dict(stats["tool_usage"])
    stats["hourly_distribution"] = dict(stats["hourly_distribution"])

    return stats


def make_json_serializable(obj: Any) -> Any:
    """Convert sets and other non-JSON serializable objects to JSON-compatible types."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def process_entry_global(
    data: Dict[str, Any], stats: Dict[str, Any], session_id: str, source: str
) -> None:
    """Process a single log entry with global deduplication."""
    entry_type = data.get("type")

    if entry_type == "user" and "cwd" in data:
        # Track user messages and cwd
        session_stats = stats["by_session"][session_id]
        session_stats["user_messages"] += 1
        if not session_stats["cwd"]:
            session_stats["cwd"] = data["cwd"]

        # Update timestamps
        if "timestamp" in data:
            timestamp = data["timestamp"]
            if (
                not session_stats["start_time"]
                or timestamp < session_stats["start_time"]
            ):
                session_stats["start_time"] = timestamp
            if not session_stats["end_time"] or timestamp > session_stats["end_time"]:
                session_stats["end_time"] = timestamp

    elif entry_type == "assistant" and "message" in data:
        message = data["message"]
        usage = message.get("usage", {})

        if not usage:
            return

        # Global deduplication check
        message_id = message.get("id")
        request_id = data.get("requestId")
        if message_id and request_id:
            dedup_key = f"{message_id}:{request_id}"
            if dedup_key in stats["seen_messages"]:
                # Skip duplicate message
                return
            stats["seen_messages"].add(dedup_key)

        # Extract token counts
        tokens = {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "cache_creation": usage.get("cache_creation_input_tokens", 0),
            "cache_read": usage.get("cache_read_input_tokens", 0),
        }

        # Update total stats
        stats["input_tokens"] += tokens["input"]
        stats["output_tokens"] += tokens["output"]
        stats["cache_creation_input_tokens"] += tokens["cache_creation"]
        stats["cache_read_input_tokens"] += tokens["cache_read"]
        stats["total_messages"] += 1

        # Update model stats
        model = message.get("model", "unknown")
        model_stats = stats["by_model"][model]
        model_stats["input_tokens"] += tokens["input"]
        model_stats["output_tokens"] += tokens["output"]
        model_stats["cache_creation_input_tokens"] += tokens["cache_creation"]
        model_stats["cache_read_input_tokens"] += tokens["cache_read"]
        model_stats["messages"] += 1
        if request_id:
            model_stats["requests"].add(request_id)

        # Update session stats
        session_stats = stats["by_session"][session_id]
        session_stats["input_tokens"] += tokens["input"]
        session_stats["output_tokens"] += tokens["output"]
        session_stats["cache_creation_input_tokens"] += tokens["cache_creation"]
        session_stats["cache_read_input_tokens"] += tokens["cache_read"]
        session_stats["messages"] += 1
        session_stats["models_used"].add(model)

        # Update timestamps
        if "timestamp" in data:
            timestamp = data["timestamp"]
            if (
                not session_stats["start_time"]
                or timestamp < session_stats["start_time"]
            ):
                session_stats["start_time"] = timestamp
            if not session_stats["end_time"] or timestamp > session_stats["end_time"]:
                session_stats["end_time"] = timestamp

            # Update date stats
            try:
                dt = date_parser.parse(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
                date_stats = stats["by_date"][date_str]
                date_stats["input_tokens"] += tokens["input"]
                date_stats["output_tokens"] += tokens["output"]
                date_stats["cache_creation_input_tokens"] += tokens["cache_creation"]
                date_stats["cache_read_input_tokens"] += tokens["cache_read"]
                date_stats["messages"] += 1
                date_stats["sessions"].add(session_id)

                # Update hourly distribution
                hour = dt.hour
                stats["hourly_distribution"][hour] += 1
            except Exception:
                pass

        # Track tool usage
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_name = item.get("name", "unknown")
                    stats["tool_usage"][tool_name] += 1

        # Store message for analytics
        stats["all_messages"].append(
            {
                "timestamp": data.get("timestamp"),
                "model": model,
                "usage": usage,
                "role": "assistant",
                "session_id": session_id,
                "source": source,
            }
        )


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def format_cost(cost: float) -> str:
    """Format cost with dollar sign and appropriate decimal places."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_duration(start: str, end: str) -> str:
    """Format duration between two timestamps."""
    try:
        start_dt = date_parser.parse(start)
        end_dt = date_parser.parse(end)
        duration = end_dt - start_dt

        days = duration.days
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or not parts:
            parts.append(f"{minutes}m")

        return " ".join(parts)
    except Exception:
        return "N/A"


def find_docker_claude_dirs() -> List[Tuple[str, str, str]]:
    """Find Claude directories in Docker containers."""
    docker_dirs = []

    try:
        # Get all containers (including stopped ones)
        result = subprocess.run(
            ["docker", "ps", "-a", "-q"], capture_output=True, text=True, check=True
        )
        container_ids = result.stdout.strip().split("\n")

        for container_id in container_ids:
            if not container_id:
                continue

            # Get container info including state and mounts
            inspect_result = subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True,
                text=True,
                check=True,
            )
            container_info = json.loads(inspect_result.stdout)[0]
            container_name = container_info["Name"].lstrip("/")
            is_running = container_info["State"]["Running"]

            # For running containers, use docker exec
            if is_running:
                # Find .claude directories in the container
                find_result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        container_id,
                        "sh",
                        "-c",
                        "find / -maxdepth 4 -name .claude -type d 2>/dev/null || true",
                    ],
                    capture_output=True,
                    text=True,
                )

                # Check both stdout and stderr
                output = find_result.stdout or find_result.stderr
                if output:
                    claude_paths = output.strip().split("\n")
                    for claude_path in claude_paths:
                        if claude_path and "/.claude" in claude_path:
                            claude_path = claude_path.strip()
                            # Skip system paths
                            if (
                                "/proc/" not in claude_path
                                and "/sys/" not in claude_path
                            ):
                                # Determine location type
                                if "/root/" in claude_path:
                                    location = "root"
                                elif "/home/" in claude_path:
                                    parts = claude_path.split("/")
                                    if len(parts) >= 3 and parts[1] == "home":
                                        location = parts[2]
                                    else:
                                        location = "home"
                                else:
                                    location = "other"

                                docker_dirs.append(
                                    (
                                        f"{container_name} ({container_id[:12]}) [{location}]",
                                        container_id,
                                        claude_path,
                                    )
                                )
            else:
                # For stopped containers, check for volume mounts
                mounts = container_info.get("Mounts", [])
                for mount in mounts:
                    if mount.get("Type") in ["bind", "volume"]:
                        destination = mount.get("Destination", "")
                        # Check if this mount might contain .claude
                        if "/.claude" in destination or destination.endswith(
                            "/.claude"
                        ):
                            # Determine location type
                            if "/root/" in destination:
                                location = "root"
                            elif "/home/" in destination:
                                parts = destination.split("/")
                                if len(parts) >= 3 and parts[1] == "home":
                                    location = parts[2]
                                else:
                                    location = "home"
                            else:
                                location = "other"

                            docker_dirs.append(
                                (
                                    f"{container_name} ({container_id[:12]}) [stopped,{location}]",
                                    container_id,
                                    destination,
                                )
                            )

    except subprocess.CalledProcessError:
        pass

    return docker_dirs


def show_help(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Custom help callback that displays help in a Rich panel."""
    if not value or ctx.resilient_parsing:
        return

    # Build help content manually for better formatting
    help_content = []

    # Usage line
    help_content.append("[bold]Usage:[/bold] blackbox [OPTIONS]")
    help_content.append("")

    # Description
    help_content.append("[bold]Description:[/bold]")
    help_content.append("  Analyse AirPilot usage.")
    help_content.append("")

    # Options
    help_content.append("[bold]Options:[/bold]")

    options_info = [
        ("--claude-dir TEXT", "Path to Claude directory (default: ~/.claude)"),
        ("--docker", "Search for Claude directories in Docker containers"),
        ("--no-litellm", "Disable LiteLLM pricing (use hardcoded prices)"),
        ("--refresh-pricing", "Force refresh of LiteLLM pricing data"),
        ("-o, --output PATH", "Export results to JSON file"),
        ("--summary-only", "Show only the summary statistics"),
        ("-t, --tools", "Show tool usage statistics"),
        ("-l, --limit INTEGER", "Limit number of items shown in tables (default: 10)"),
        ("--cache", "Show cache efficiency analytics"),
        ("--response-times", "Show response time analysis"),
        ("--full", "Show all analytics including cache, response times, and tools"),
        ("--help", "Show this message and exit"),
    ]

    for option, description in options_info:
        help_content.append(f"  [cyan]{option:<25}[/cyan] {description}")

    # Create a rich panel with the help content
    panel = Panel(
        "\n".join(help_content),
        title="[bold cyan]AirPilot BlackBox Help[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )

    console.print(panel)
    ctx.exit()


def show_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Callback to show version information."""
    if not value or ctx.resilient_parsing:
        return

    try:
        pkg_version = get_version("airpilot-blackbox")
    except Exception:
        pkg_version = "unknown"

    version_info = f"""[bold cyan]AirPilot BlackBox[/bold cyan]

Version: [bold green]{pkg_version}[/bold green]
Package: airpilot-blackbox
CLI Aliases: blackbox, airpilot-blackbox

[dim]Analyze Claude AI usage logs and calculate costs[/dim]"""

    panel = Panel(
        version_info,
        title="[bold cyan]Version Information[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )

    console.print(panel)
    ctx.exit()


@click.command(context_settings={"help_option_names": []})
@click.option(
    "--help",
    "-h",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=show_help,
    help="Show this message and exit",
)
@click.option(
    "--version",
    "-v",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=show_version,
    help="Show version information",
)
@click.option(
    "--claude-dir",
    default="~/.claude",
    help="Path to Claude directory (default: ~/.claude)",
)
@click.option(
    "--docker", is_flag=True, help="Search for Claude directories in Docker containers"
)
@click.option(
    "--no-litellm", is_flag=True, help="Disable LiteLLM pricing (use hardcoded prices)"
)
@click.option(
    "--refresh-pricing", is_flag=True, help="Force refresh of LiteLLM pricing data"
)
@click.option("--output", "-o", type=click.Path(), help="Export results to JSON file")
@click.option("--summary-only", is_flag=True, help="Show only the summary statistics")
@click.option("--tools", "-t", is_flag=True, help="Show tool usage statistics")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=10,
    help="Limit number of items shown in tables (default: 10)",
)
@click.option("--cache", is_flag=True, help="Show cache efficiency analytics")
@click.option("--response-times", is_flag=True, help="Show response time analysis")
@click.option(
    "--full",
    is_flag=True,
    help="Show all analytics including cache, response times, and tools",
)
def cli(
    claude_dir: str,
    docker: bool,
    no_litellm: bool,
    refresh_pricing: bool,
    output: Optional[str],
    summary_only: bool,
    tools: bool,
    limit: int,
    cache: bool,
    response_times: bool,
    full: bool,
) -> None:
    """AirPilot BlackBox - Analyze Claude AI usage logs and calculate costs."""
    # This is now the main analysis function
    analyze_impl(
        claude_dir,
        docker,
        no_litellm,
        refresh_pricing,
        output,
        summary_only,
        tools,
        limit,
        cache,
        response_times,
        full,
    )


def analyze_impl(
    claude_dir: str,
    docker: bool,
    no_litellm: bool,
    refresh_pricing: bool,
    output: Optional[str],
    summary_only: bool,
    tools: bool,
    limit: int,
    cache: bool,
    response_times: bool,
    full: bool,
) -> None:
    """Analyse AirPilot usage."""

    # Handle pricing refresh if requested
    if refresh_pricing and not no_litellm:
        try:
            from .pricing_fetcher import LiteLLMPricingFetcher

            fetcher = LiteLLMPricingFetcher()
            fetcher.fetch_pricing(force_refresh=True)
            console.print("[green]✓[/green] LiteLLM pricing data refreshed")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to refresh pricing: {e}[/yellow]")

    console.print(
        Panel.fit(
            "[bold cyan]AirPilot Blackbox[/bold cyan]\nAnalyse AirPilot usage",
            border_style="cyan",
        )
    )

    # Collect all JSONL files from all sources for unified parsing
    all_jsonl_files = []
    sources = []

    # Collect local files
    claude_path = Path(claude_dir).expanduser()
    if (claude_path / "projects").exists():
        local_files = list((claude_path / "projects").rglob("*.jsonl"))
        if local_files:
            all_jsonl_files.extend([(f, "local") for f in local_files])
            sources.append("local")

    # Collect Docker files if requested
    docker_temp_dirs = []
    if docker:
        docker_containers = find_docker_claude_dirs()
        if docker_containers:
            sources.append("docker")
            for (
                container_name,
                container_id,
                container_claude_path,
            ) in docker_containers:
                # Copy to temp directory
                temp_dir = tempfile.mkdtemp()
                docker_temp_dirs.append(temp_dir)
                temp_claude = Path(temp_dir) / ".claude"

                console.print(
                    f"Copying {container_claude_path} from {container_name}..."
                )
                copy_result = subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"{container_id}:{container_claude_path}",
                        str(temp_claude),
                    ],
                    capture_output=True,
                    text=True,
                )

                if copy_result.returncode == 0:
                    docker_files = list((temp_claude / "projects").rglob("*.jsonl"))
                    all_jsonl_files.extend(
                        [(f, f"docker:{container_name}") for f in docker_files]
                    )

    # Parse all files with global deduplication
    if all_jsonl_files:
        console.print(
            f"\nFound {len(all_jsonl_files)} JSONL files across {len(sources)} source(s)"
        )

        # Create a unified parser with global deduplication
        merged_stats = parse_all_files_globally(all_jsonl_files)

        # Calculate costs
        calculator = CostCalculator(use_litellm=not no_litellm)
        merged_costs = calculator.calculate_costs(merged_stats)

        # Export if requested
        if output:
            if summary_only:
                # Create summary-only export with all summary tables
                summary_data = {
                    "overall_stats": {
                        "total_messages": merged_stats.get("total_messages", 0),
                        "input_tokens": merged_stats.get("input_tokens", 0),
                        "output_tokens": merged_stats.get("output_tokens", 0),
                        "cache_creation_tokens": merged_stats.get(
                            "cache_creation_input_tokens", 0
                        ),
                        "cache_read_tokens": merged_stats.get(
                            "cache_read_input_tokens", 0
                        ),
                        "total_tokens": (
                            merged_stats.get("input_tokens", 0)
                            + merged_stats.get("output_tokens", 0)
                            + merged_stats.get("cache_creation_input_tokens", 0)
                            + merged_stats.get("cache_read_input_tokens", 0)
                        ),
                        "total_cost": merged_costs.get("total", {}).get("total_cost", 0),
                    },
                    "model_breakdown": merged_costs.get("by_model", {}),
                    "daily_breakdown": merged_costs.get("by_date", {}),
                    "session_breakdown": merged_costs.get("by_session", {}),
                    "sources": sources,
                }
                json_data = make_json_serializable(summary_data)
            else:
                # Full export
                json_data = make_json_serializable(
                    {"stats": merged_stats, "costs": merged_costs, "sources": sources}
                )

            with open(output, "w") as f:
                json.dump(json_data, f, indent=2)
            console.print(f"\n[green]✓[/green] Results exported to {output}")

        # Display results with source indicators
        display_results(
            merged_stats,
            merged_costs,
            sources,
            summary_only,
            tools,
            limit,
            cache,
            response_times,
            full,
        )

        # Clean up Docker temp directories
        for temp_dir in docker_temp_dirs:
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass
    else:
        console.print("[red]No Claude usage data found[/red]")


def analyze_claude_dir_raw(
    claude_dir: str, use_litellm: bool = True
) -> Tuple[Optional[dict], Optional[dict]]:
    """Analyze a Claude directory and return raw stats and costs."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing local usage logs...", total=None)

            parser = UsageParser(claude_dir)
            stats = parser.parse_all_logs()

            if not stats.get("total_messages"):
                return None, None

            progress.update(task, description="Calculating costs...")
            calculator = CostCalculator(use_litellm=use_litellm)
            costs = calculator.calculate_costs(stats)

            progress.update(task, completed=True)

        return stats, costs
    except Exception as e:
        console.print(
            f"[yellow]Warning: Failed to analyze local directory: {e}[/yellow]"
        )
        return None, None


def handle_docker_analysis_raw(
    use_litellm: bool = True,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Handle Docker container analysis and return raw stats and costs."""
    console.print(
        "\n[yellow]Searching for Claude directories in Docker containers...[/yellow]"
    )
    docker_containers = find_docker_claude_dirs()

    if not docker_containers:
        console.print(
            "[yellow]No Claude directories found in Docker containers[/yellow]"
        )
        return None, None

    console.print(
        f"Found {len(docker_containers)} Claude directories in Docker containers:"
    )
    for i, (container_info, _, claude_path) in enumerate(docker_containers):
        console.print(f"  [{i + 1}] {container_info} - {claude_path}")

    # Analyze all containers
    all_docker_stats = []
    all_docker_costs = []

    for container_name, container_id, container_claude_path in docker_containers:
        console.print(f"\nAnalyzing: {container_name}")

        # Copy .claude directory from container to temp location
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_claude = Path(temp_dir) / ".claude"

            console.print(f"Copying {container_claude_path} from container...")
            copy_result = subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container_id}:{container_claude_path}",
                    str(temp_claude),
                ],
                capture_output=True,
                text=True,
            )

            if copy_result.returncode == 0:
                # Analyze the copied directory
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Parsing {container_name} usage logs...", total=None
                    )

                    parser = UsageParser(str(temp_claude))
                    stats = parser.parse_all_logs()

                    if stats.get("total_messages"):
                        progress.update(task, description="Calculating costs...")
                        calculator = CostCalculator(use_litellm=use_litellm)
                        costs = calculator.calculate_costs(stats)

                        all_docker_stats.append(stats)
                        all_docker_costs.append(costs)

                    progress.update(task, completed=True)
            else:
                console.print(
                    f"[red]Failed to copy .claude directory: {copy_result.stderr}[/red]"
                )

    # Merge all Docker container results
    if all_docker_stats:
        merged_stats, merged_costs = merge_stats_and_costs(
            all_docker_stats, all_docker_costs
        )
        return merged_stats, merged_costs
    else:
        return None, None


def merge_stats_and_costs(
    all_stats: List[Dict[str, Any]], all_costs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge multiple stats and costs dictionaries with global deduplication."""
    merged_stats: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_messages": 0,
        "by_model": {},
        "by_session": {},
        "by_date": {},
        "errors": [],
        "tool_usage": {},
        "hourly_distribution": {},
        "all_messages": [],  # Store all messages for analytics
        "seen_messages": set(),  # Track globally seen messages for deduplication
    }

    merged_costs: Dict[str, Any] = {
        "by_model": {},
        "by_session": {},
        "by_date": {},
        "total": {
            "input_cost": 0,
            "output_cost": 0,
            "cache_write_cost": 0,
            "cache_read_cost": 0,
            "total_cost": 0,
        },
    }

    # Merge stats
    for i, stats in enumerate(all_stats):  # noqa: B007
        # Merge by_model (this is already deduplicated at model level)
        for model, model_stats in stats.get("by_model", {}).items():
            if model not in merged_stats["by_model"]:
                merged_stats["by_model"][model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "messages": 0,
                    "requests": [],
                }
            for key in [
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "messages",
            ]:
                merged_stats["by_model"][model][key] += model_stats.get(key, 0)
            merged_stats["by_model"][model]["requests"].extend(
                model_stats.get("requests", [])
            )

        # Merge sessions - for now just collect all sessions
        for session_id, session_stats in stats.get("by_session", {}).items():
            if session_id not in merged_stats["by_session"]:
                merged_stats["by_session"][session_id] = session_stats
            else:
                # Merge session data - this will be recalculated later
                existing = merged_stats["by_session"][session_id]
                existing["messages"] = existing.get("messages", 0) + session_stats.get(
                    "messages", 0
                )

        # Merge by_date
        for date, date_stats in stats.get("by_date", {}).items():
            if date not in merged_stats["by_date"]:
                merged_stats["by_date"][date] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "messages": 0,
                    "sessions": [],
                }
            for key in [
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "messages",
            ]:
                merged_stats["by_date"][date][key] += date_stats.get(key, 0)
            merged_stats["by_date"][date]["sessions"].extend(
                date_stats.get("sessions", [])
            )

        # Merge tool usage
        for tool, count in stats.get("tool_usage", {}).items():
            merged_stats["tool_usage"][tool] = (
                merged_stats["tool_usage"].get(tool, 0) + count
            )

        # Merge errors
        merged_stats["errors"].extend(stats.get("errors", []))

        # Merge all messages
        merged_stats["all_messages"].extend(stats.get("all_messages", []))

    # After merging, recalculate totals from deduplicated sessions
    merged_stats["input_tokens"] = 0
    merged_stats["output_tokens"] = 0
    merged_stats["cache_creation_input_tokens"] = 0
    merged_stats["cache_read_input_tokens"] = 0
    merged_stats["total_messages"] = 0

    for session_stats in merged_stats["by_session"].values():
        merged_stats["input_tokens"] += session_stats.get("input_tokens", 0)
        merged_stats["output_tokens"] += session_stats.get("output_tokens", 0)
        merged_stats["cache_creation_input_tokens"] += session_stats.get(
            "cache_creation_input_tokens", 0
        )
        merged_stats["cache_read_input_tokens"] += session_stats.get(
            "cache_read_input_tokens", 0
        )
        merged_stats["total_messages"] += session_stats.get("messages", 0)

    # Merge costs
    for i, costs in enumerate(all_costs):
        # Merge by_model costs (keep as is - model level is accurate)
        for model, model_costs in costs.get("by_model", {}).items():
            if model not in merged_costs["by_model"]:
                merged_costs["by_model"][model] = {
                    "input_cost": 0,
                    "output_cost": 0,
                    "cache_write_cost": 0,
                    "cache_read_cost": 0,
                    "total_cost": 0,
                }
            for key in [
                "input_cost",
                "output_cost",
                "cache_write_cost",
                "cache_read_cost",
                "total_cost",
            ]:
                merged_costs["by_model"][model][key] += model_costs.get(key, 0)

        # Merge session costs - keep matching the stats deduplication
        for session_id, session_cost in costs.get("by_session", {}).items():
            # Check if this session is in merged_stats (it should be if we kept it)
            if session_id in merged_stats["by_session"]:
                # Check if this cost data matches the stats data we kept
                stats_messages = merged_stats["by_session"][session_id].get(
                    "messages", 0
                )
                cost_messages = (
                    all_stats[i]
                    .get("by_session", {})
                    .get(session_id, {})
                    .get("messages", 0)
                )
                if stats_messages == cost_messages:
                    merged_costs["by_session"][session_id] = session_cost

        # Merge date costs
        for date, date_costs in costs.get("by_date", {}).items():
            if date not in merged_costs["by_date"]:
                merged_costs["by_date"][date] = {
                    "input_cost": 0,
                    "output_cost": 0,
                    "cache_write_cost": 0,
                    "cache_read_cost": 0,
                    "total_cost": 0,
                }
            for key in [
                "input_cost",
                "output_cost",
                "cache_write_cost",
                "cache_read_cost",
                "total_cost",
            ]:
                merged_costs["by_date"][date][key] += date_costs.get(key, 0)

    # Recalculate total costs from deduplicated sessions
    merged_costs["total"] = {
        "input_cost": 0,
        "output_cost": 0,
        "cache_write_cost": 0,
        "cache_read_cost": 0,
        "total_cost": 0,
    }

    for session_cost in merged_costs["by_session"].values():
        for key in merged_costs["total"]:
            merged_costs["total"][key] += session_cost.get(key, 0)

    return merged_stats, merged_costs


def display_results(
    stats: Dict[str, Any],
    costs: Dict[str, Any],
    sources: List[str],
    summary_only: bool,
    tools: bool,
    limit: int,
    cache: bool = False,
    response_times: bool = False,
    full: bool = False,
) -> None:
    """Display merged results with source indicators."""
    # Add source indicator to title if includes Docker
    source_text = ""
    if len(sources) > 1:
        source_text = " (local + docker)"
    elif "docker" in sources:
        source_text = " (docker only)"

    # Display overall statistics
    display_overall_stats(stats, costs, source_text)

    if not summary_only:
        display_model_breakdown(stats, costs, limit)
        display_daily_breakdown(stats, costs, limit)
        display_session_breakdown(stats, costs, limit)

        # Display enhanced analytics
        if full or cache:
            display_cache_analytics(stats)

        if full or response_times:
            display_response_time_analysis(stats)

        if full:
            display_enhanced_tool_usage(stats, limit)
        elif tools:
            display_tool_usage(stats, limit)

    # Display errors if any
    if stats.get("errors"):
        console.print(
            f"\n[yellow]⚠[/yellow] Found {len(stats['errors'])} errors while parsing logs"
        )


def display_overall_stats(
    stats: Dict[str, Any], costs: Dict[str, Any], source_text: str = ""
) -> None:
    """Display overall usage statistics."""
    total_tokens = (
        stats["input_tokens"]
        + stats["output_tokens"]
        + stats["cache_creation_input_tokens"]
        + stats["cache_read_input_tokens"]
    )

    title = f"[bold white]Overall Usage Statistics{source_text}[/bold white]"
    table = Table(title=title, show_header=False, box=None, title_justify="left")
    table.add_column("Metric", style="dim cyan")
    table.add_column("Value", justify="right", style="dim white")

    table.add_row("Total Messages", format_number(stats["total_messages"]))
    table.add_row("Input Tokens", format_number(stats["input_tokens"]))
    table.add_row("Output Tokens", format_number(stats["output_tokens"]))
    table.add_row(
        "Cache Creation Tokens", format_number(stats["cache_creation_input_tokens"])
    )
    table.add_row("Cache Read Tokens", format_number(stats["cache_read_input_tokens"]))
    table.add_row("Total Tokens", f"[bold]{format_number(total_tokens)}[/bold]")
    table.add_row("", "")
    table.add_row(
        "Total Cost",
        f"[bold green]{format_cost(costs['total']['total_cost'])}[/bold green]",
    )

    console.print("\n", table)


def display_model_breakdown(
    stats: Dict[str, Any], costs: Dict[str, Any], limit: int
) -> None:
    """Display breakdown by model."""
    console.print("\n[bold]Model Usage Breakdown[/bold]")

    table = Table(show_header=True)
    table.add_column("Model", style="dim cyan")
    table.add_column("Messages", justify="right", style="dim white")
    table.add_column("Input", justify="right", style="dim white")
    table.add_column("Output", justify="right", style="dim white")
    table.add_column("Cache", justify="right", style="dim white")
    table.add_column("Cost", justify="right", style="green")

    # Sort models by total cost
    sorted_models = sorted(
        costs["by_model"].items(), key=lambda x: x[1]["total_cost"], reverse=True
    )

    for model, model_costs in sorted_models[:limit]:
        model_stats = stats["by_model"][model]
        cache_tokens = model_stats.get(
            "cache_creation_input_tokens", 0
        ) + model_stats.get("cache_read_input_tokens", 0)

        # Shorten model name for display
        display_model = model
        if len(model) > 30:
            display_model = model[:27] + "..."

        table.add_row(
            display_model,
            format_number(model_stats["messages"]),
            format_number(model_stats["input_tokens"]),
            format_number(model_stats["output_tokens"]),
            format_number(cache_tokens),
            format_cost(model_costs["total_cost"]),
        )

    console.print(table)


def display_daily_breakdown(
    stats: Dict[str, Any], costs: Dict[str, Any], limit: int
) -> None:
    """Display daily usage breakdown."""
    console.print("\n[bold]Daily Usage[/bold]")

    table = Table(show_header=True)
    table.add_column("Date", style="dim cyan")
    table.add_column("Messages", justify="right", style="dim white")
    table.add_column("Sessions", justify="right", style="dim white")
    table.add_column("Tokens", justify="right", style="dim white")
    table.add_column("Cost", justify="right", style="green")

    # Sort dates
    sorted_dates = sorted(costs["by_date"].items(), reverse=True)

    for date, date_costs in sorted_dates[:limit]:
        date_stats = stats["by_date"][date]
        total_tokens = (
            date_stats.get("input_tokens", 0)
            + date_stats.get("output_tokens", 0)
            + date_stats.get("cache_creation_input_tokens", 0)
            + date_stats.get("cache_read_input_tokens", 0)
        )

        table.add_row(
            date,
            format_number(date_stats["messages"]),
            str(len(date_stats.get("sessions", []))),
            format_number(total_tokens),
            format_cost(date_costs["total_cost"]),
        )

    console.print(table)


def display_session_breakdown(
    stats: Dict[str, Any], costs: Dict[str, Any], limit: int
) -> None:
    """Display breakdown by session."""
    console.print("\n[bold]Session Breakdown[/bold]")

    table = Table(show_header=True)
    table.add_column("Session ID", style="dim cyan")
    table.add_column("Messages", justify="right", style="dim white")
    table.add_column("Duration", justify="right", style="dim white")
    table.add_column("Models", justify="left", style="dim white")
    table.add_column("Cost", justify="right", style="green")

    # Sort sessions by total cost
    sorted_sessions = sorted(
        costs["by_session"].items(), key=lambda x: x[1]["total_cost"], reverse=True
    )

    for session_id, session_costs in sorted_sessions[:limit]:
        session_stats = stats["by_session"][session_id]

        # Format models used
        models = list(session_stats.get("models_used", set()))
        if models:
            # Shorten model names
            short_models = []
            for m in models[:2]:
                if "-" in m:
                    parts = m.split("-")
                    if len(parts) > 2:
                        short_models.append(parts[1])
                    else:
                        short_models.append(m)
                else:
                    short_models.append(m)
            models_str = ", ".join(short_models)
            if len(models) > 2:
                models_str += f", +{len(models) - 2}"
        else:
            models_str = ""

        table.add_row(
            session_id[:8] + "...",
            format_number(session_stats["messages"]),
            format_duration(
                session_stats.get("start_time", ""), session_stats.get("end_time", "")
            ),
            models_str,
            format_cost(session_costs["total_cost"]),
        )

    console.print(table)


def display_tool_usage(stats: Dict[str, Any], limit: int) -> None:
    """Display tool usage statistics."""
    console.print("\n[bold]Tool Usage[/bold]")

    table = Table(show_header=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Usage Count", justify="right")
    table.add_column("Percentage", justify="right")

    tool_usage = stats.get("tool_usage", {})
    total_tool_uses = sum(tool_usage.values())

    # Sort tools by usage
    sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)

    for tool, count in sorted_tools[:limit]:
        percentage = (count / total_tool_uses * 100) if total_tool_uses > 0 else 0
        table.add_row(tool, format_number(count), f"{percentage:.1f}%")

    console.print(table)


def display_cache_analytics(stats: Dict[str, Any]) -> None:
    """Display cache efficiency analytics."""
    console.print("\n[bold]Cache Analytics[/bold]")

    cache_metrics = calculate_cache_metrics(stats)

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    # Format percentages and values
    hit_rate = cache_metrics.get("cache_hit_rate", 0) * 100
    efficiency = cache_metrics.get("cache_efficiency", 0)
    roi = cache_metrics.get("cache_roi", 0)
    savings = cache_metrics.get("cache_total_savings", 0)

    table.add_row("Cache Hit Rate", f"{hit_rate:.1f}%")
    table.add_row("Cache Efficiency", f"{efficiency:.1f}x")
    table.add_row("Cache ROI", format_cost(roi))
    table.add_row("Total Savings", format_cost(savings))
    table.add_row(
        "Cache Creation Cost", format_cost(cache_metrics.get("cache_creation_cost", 0))
    )

    console.print(table)

    # Show top sessions by cache efficiency
    if "by_session" in cache_metrics:
        console.print("\n[bold]Top Sessions by Cache Efficiency[/bold]")
        session_table = Table(show_header=True)
        session_table.add_column("Session", style="cyan")
        session_table.add_column("Efficiency", justify="right")
        session_table.add_column("Cache Read", justify="right")
        session_table.add_column("Cache Write", justify="right")

        sorted_sessions = sorted(
            cache_metrics["by_session"].items(),
            key=lambda x: x[1]["efficiency"],
            reverse=True,
        )[:5]

        for session_id, metrics in sorted_sessions:
            if metrics["efficiency"] > 0:
                session_table.add_row(
                    session_id[:8] + "...",
                    f"{metrics['efficiency']:.1f}x",
                    format_number(metrics["cache_read"]),
                    format_number(metrics["cache_write"]),
                )

        if sorted_sessions:
            console.print(session_table)


def display_response_time_analysis(stats: Dict[str, Any]) -> None:
    """Display response time analysis."""
    console.print("\n[bold]Response Time Analysis[/bold]")

    # Calculate response times from all messages
    all_messages = stats.get("all_messages", [])
    if not all_messages:
        console.print(
            "[yellow]No message data available for response time analysis[/yellow]"
        )
        return

    response_metrics = calculate_response_times(all_messages)

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Time (s)", justify="right")

    table.add_row("Average Response", f"{response_metrics['avg_response_time']:.1f}")
    table.add_row("Median Response", f"{response_metrics['median_response_time']:.1f}")
    table.add_row("95th Percentile", f"{response_metrics['p95_response_time']:.1f}")
    table.add_row("99th Percentile", f"{response_metrics['p99_response_time']:.1f}")
    table.add_row("Fastest Response", f"{response_metrics['min_response_time']:.1f}")
    table.add_row("Slowest Response", f"{response_metrics['max_response_time']:.1f}")
    table.add_row("Total Responses", format_number(response_metrics["total_responses"]))

    console.print(table)

    # By model breakdown
    if response_metrics["by_model"]:
        console.print("\n[bold]Response Times by Model[/bold]")
        model_table = Table(show_header=True)
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Avg", justify="right")
        model_table.add_column("Median", justify="right")
        model_table.add_column("P95", justify="right")
        model_table.add_column("Count", justify="right")

        for model, metrics in sorted(response_metrics["by_model"].items()):
            # Shorten model name
            display_model = model
            if len(model) > 25:
                display_model = model[:22] + "..."

            model_table.add_row(
                display_model,
                f"{metrics['avg']:.1f}s",
                f"{metrics['median']:.1f}s",
                f"{metrics['p95']:.1f}s",
                format_number(metrics["count"]),
            )

        console.print(model_table)


def display_enhanced_tool_usage(stats: Dict[str, Any], limit: int) -> None:
    """Display enhanced tool usage analytics."""
    console.print("\n[bold]Tool Usage Analytics[/bold]")

    all_messages = stats.get("all_messages", [])
    if not all_messages:
        console.print("[yellow]No message data available for tool analysis[/yellow]")
        return

    tool_analytics = analyze_tool_usage(all_messages)

    table = Table(show_header=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Cost/Use", justify="right")

    tools_data = tool_analytics.get("tools", {})
    for tool, data in list(tools_data.items())[:limit]:
        table.add_row(
            tool,
            format_number(data["count"]),
            format_number(int(data["avg_tokens"])),
            format_cost(data["total_cost"]),
            format_cost(data["avg_cost"]),
        )

    console.print(table)

    # Show tool combinations
    any_combinations = any(data["top_combinations"] for data in tools_data.values())
    if any_combinations:
        console.print("\n[bold]Top Tool Combinations[/bold]")
        combo_table = Table(show_header=True)
        combo_table.add_column("Tool Pair", style="cyan")
        combo_table.add_column("Count", justify="right")

        # Collect all combinations
        all_combos = {}
        for tool, data in tools_data.items():
            for combo_tool, count in data["top_combinations"]:
                pair = tuple(sorted([tool, combo_tool]))
                if pair not in all_combos:
                    all_combos[pair] = count

        # Sort and display top combinations
        sorted_combos = sorted(all_combos.items(), key=lambda x: x[1], reverse=True)[:5]
        for (tool1, tool2), count in sorted_combos:
            combo_table.add_row(f"{tool1} + {tool2}", format_number(count))

        if sorted_combos:
            console.print(combo_table)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
