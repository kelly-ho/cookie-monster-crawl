"""
General-purpose investigation tools for strategy analysis.

The LLM requests investigations by specifying a tool name and arguments.
This module executes those requests and returns plain-text findings.
"""

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent
MAX_RESULT_CHARS = 4000

TOOL_DESCRIPTIONS = {
    "fetch_url": {
        "description": "Fetch a URL and report HTTP status, JSON-LD schema types found, page title, and link count.",
        "args": {"url": "<URL to fetch>"},
    },
    "query_log": {
        "description": "Query the JSONL crawl log for events matching filters. 'domain' matches as substring against the url field. All other filter keys match exactly.",
        "args": {"filter": {"type": "<event type>", "domain": "<substring>", "<key>": "<value>"}, "limit": "<int, default 10>"},
    },
    "read_file": {
        "description": "Read a project source file. If 'search' is provided, return only matching lines with context.",
        "args": {"path": "<relative path from project root>", "search": "<optional grep string>"},
    },
    "list_urls": {
        "description": "List URLs from the crawl log by domain and status.",
        "args": {"domain": "<domain substring>", "status": "<recipe|non_recipe|filtered|visited>", "limit": "<int, default 10>"},
    },
}


def fetch_url(args: dict) -> str:
    """Fetch a URL and summarize what's there."""
    url = args.get("url", "")
    if not url:
        return "Error: no URL provided"

    try:
        req = Request(url, headers={"User-Agent": "CookieMonsterCrawler/0.1"})
        resp = urlopen(req, timeout=10)
        status = resp.status
        content_type = resp.headers.get("Content-Type", "unknown")
        html = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        return f"HTTP {e.code}: {e.reason}"
    except (URLError, TimeoutError, Exception) as e:
        return f"Error: {e}"

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "(no title)"

    # JSON-LD analysis
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    json_ld_types = []
    has_recipe = False
    for script in scripts:
        try:
            data = json.loads(script.get_text(strip=True))
            types = _collect_types(data)
            json_ld_types.extend(types)
            if "Recipe" in types:
                has_recipe = True
        except (json.JSONDecodeError, TypeError):
            continue

    # Link count
    links = soup.find_all("a", href=True)
    base_domain = urlparse(url).netloc
    same_domain_links = sum(1 for a in links if urlparse(a["href"]).netloc in ("", base_domain))

    lines = [
        f"URL: {url}",
        f"HTTP Status: {status}",
        f"Content-Type: {content_type}",
        f"Title: {title}",
        f"JSON-LD blocks: {len(scripts)}",
        f"JSON-LD types: {json_ld_types or '(none)'}",
        f"Has Recipe schema: {has_recipe}",
        f"Total links: {len(links)}",
        f"Same-domain links: {same_domain_links}",
    ]
    return "\n".join(lines)


def _collect_types(data) -> list[str]:
    """Recursively collect @type values from JSON-LD."""
    types = []
    if isinstance(data, dict):
        t = data.get("@type")
        if isinstance(t, list):
            types.extend(t)
        elif t:
            types.append(t)
        for v in data.values():
            types.extend(_collect_types(v))
    elif isinstance(data, list):
        for item in data:
            types.extend(_collect_types(item))
    return types


def query_log(args: dict, logfile: str) -> str:
    """Filter JSONL crawl log events."""
    filters = args.get("filter", {})
    limit = int(args.get("limit", 10))
    domain_filter = filters.pop("domain", None)

    results = []
    try:
        with open(logfile, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)

                if domain_filter and domain_filter not in event.get("url", ""):
                    continue

                if all(event.get(k) == v for k, v in filters.items()):
                    results.append(event)
                    if len(results) >= limit:
                        break
    except FileNotFoundError:
        return f"Error: log file not found: {logfile}"

    if not results:
        return "No matching events found."

    lines = [f"Found {len(results)} matching events:"]
    for event in results:
        lines.append(json.dumps(event, default=str))
    return _truncate("\n".join(lines))


def read_file(args: dict, project_root: Path = PROJECT_ROOT) -> str:
    """Read a project file, optionally grep for a string."""
    rel_path = args.get("path", "")
    search = args.get("search")

    if not rel_path:
        return "Error: no path provided"

    # Prevent directory traversal
    resolved = (project_root / rel_path).resolve()
    if not str(resolved).startswith(str(project_root.resolve())):
        return "Error: path outside project root"

    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {rel_path}"

    if search:
        lines = text.splitlines()
        matches = []
        for i, line in enumerate(lines):
            if search.lower() in line.lower():
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                for j in range(start, end):
                    prefix = ">>>" if j == i else "   "
                    matches.append(f"{prefix} {j+1}: {lines[j]}")
                matches.append("")
        return _truncate("\n".join(matches)) if matches else f"No matches for '{search}' in {rel_path}"

    lines = text.splitlines()
    if len(lines) > 100:
        return _truncate("\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[:100])) + f"\n... ({len(lines)} total lines)")
    return _truncate("\n".join(f"{i+1}: {l}" for i, l in enumerate(lines)))


def list_urls(args: dict, logfile: str) -> str:
    """List URLs from crawl log by domain and status."""
    from cookie_monster_crawl.replay import load_events, reconstruct

    domain_filter = args.get("domain")
    status = args.get("status", "visited")
    limit = int(args.get("limit", 10))

    try:
        events = load_events(logfile)
    except FileNotFoundError:
        return f"Error: log file not found: {logfile}"

    lifecycles = reconstruct(events)
    results = []

    for lc in lifecycles.values():
        if domain_filter and domain_filter not in lc.domain:
            continue

        match = False
        if status == "recipe" and lc.visited and lc.is_recipe:
            match = True
        elif status == "non_recipe" and lc.visited and lc.is_recipe is False:
            match = True
        elif status == "filtered" and lc.filtered:
            match = True
        elif status == "visited" and lc.visited:
            match = True

        if match:
            info = f"{lc.url}  score={lc.discovered_score}"
            if lc.is_recipe is not None:
                info += f"  recipe={lc.is_recipe}"
            if lc.filter_reason:
                info += f"  filter={lc.filter_reason}"
            results.append(info)
            if len(results) >= limit:
                break

    if not results:
        return f"No URLs matching domain={domain_filter} status={status}"

    return _truncate("\n".join(results))


def _truncate(text: str) -> str:
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + "\n... (truncated)"
    return text


# --- Executor ---

TOOLS = {
    "fetch_url": lambda args, **ctx: fetch_url(args),
    "query_log": lambda args, **ctx: query_log(args, ctx["logfile"]),
    "read_file": lambda args, **ctx: read_file(args, ctx.get("project_root", PROJECT_ROOT)),
    "list_urls": lambda args, **ctx: list_urls(args, ctx["logfile"]),
}


def execute(investigations: list[dict], logfile: str, project_root: Path = PROJECT_ROOT, max_fetches: int = 5) -> dict[str, str]:
    """Execute a list of investigations, return {id: result_text}."""
    findings = {}
    fetch_count = 0

    for inv in investigations:
        inv_id = inv.get("id", "unknown")
        tool = inv.get("tool", "")
        args = inv.get("args", {})

        handler = TOOLS.get(tool)
        if not handler:
            findings[inv_id] = f"Error: unknown tool '{tool}'"
            continue

        if tool == "fetch_url":
            fetch_count += 1
            if fetch_count > max_fetches:
                findings[inv_id] = f"Skipped: fetch limit ({max_fetches}) reached"
                continue

        try:
            findings[inv_id] = handler(args, logfile=logfile, project_root=project_root)
        except Exception as e:
            findings[inv_id] = f"Error: {e}"

    return findings
