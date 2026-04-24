# Cookie Monster Crawl

A self improving recipe web crawler. Uses a learned scoring model and multi agent debate to get better after every crawl.

The crawler discovers recipe pages across 50+ food websites with 96%+ accuracy. It logs every scoring decision so each run can be replayed offline. A Proposer agent analyzes the replay, investigates problems, and drafts a strategy. A Critic agent stress-tests it with evidence. The revised strategy is applied and the model is retrained.

## The Problem

Recipe websites bury content behind category pages, author profiles, and editorial sections. A crawler that follows links blindly wastes its budget on pages that don't contain recipes. The challenge is deciding which links to follow to maximize the ratio of recipe pages found per page fetched.

## The Loop

Crawl > Log > Replay > Strategy > Train > Crawl

1. **Crawl** - An async crawler fetches pages from 52 seed sites, guided by a priority queue. Each discovered link is scored by a model before entering the queue. Pages that are most likely to contain recipes receive lower scores and are fetched first.

2. **Log** — Every decision is recorded to a JSONL event log: URL discovery, scoring, fetching, recipe extraction, and filtering. Any crawl can be replayed and analyzed offline.

3. **Replay** — The replay system reconstructs every page's lifecycle from the log and can simulate how a different model would have scored the same URLs.

4. **Strategy** — A multi agent system reviews the replay data and proposes improvements. A Proposer and Critic agent debate each strategy through structured rounds.
   - **Analyze** — The Proposer reviews crawl performance and identifies problems (ex. "the crawler doesn't filter author bios"), then decides what to investigate further.
   - **Investigate** — The Proposer gathers evidence by fetching live URLs, querying the crawl log and reading source code to answer its own questions.
   - **Propose** — The Proposer produces a strategy with specific changes like new scoring features and policy adjustments.
   - **Critique** — A Critic agent stress tests the proposal. It runs its own investigations to find counter examples, checks proposals against history from past cycles, and raises evidence backed objections.
   - **Revise** — The Proposer addresses each objection, either modifying the proposal or rebutting with counter evidence.
   The debate runs for configurable rounds and outputs machine readable JSON that can be applied directly.

5. **Train** — A training pipeline builds labeled data from crawl logs, trains four model types (logistic regression, random forest, gradient boosting, SVM), and saves the best one.

## How the Crawler Works

- **URL scoring** — A trained model scores URLs using 27 features extracted from URL structure, anchor text, and crawl state. No page content is needed at scoring time.

- **Async with domain isolation** — Built on aiohttp + asyncio. Per domain locks enforce rate limits while robots.txt compliance is checked before fetching.

- **Adapts during crawl** — Tracks per domain harvest rates and uses MinHash LSH to detect structurally similar non recipe URLs. A batch rescore after seed pages are visited reprioritizes the entire queue with real domain statistics.

- **Recipe extraction** — Parses JSON-LD and microdata Recipe schemas to extract structured data (ex. title, ingredients, instructions).

- **Dynamic domain cap** — High yield domains get more queue slots, low yield domains get fewer.

## Results

96.5% mean harvest efficiency across 5 runs of 1,000 pages each, crawling 51 seed sites.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Crawl

```bash
python -m cookie_monster_crawl.crawler --max-pages 1000 --model models/model_v34.pkl --seeds data/static-target.json
```

Options:
- `--max-pages` — Page budget (default: 100)
- `--model` — Path to trained model `.pkl` (omit for hand-tuned fallback)
- `--seeds` — Seed URL list (default: `data/static-target.json`)
- `--domain-stats` — Warm-start with domain stats from a previous crawl
- `--domain-cap` — Max URLs per domain in queue (default: 50)
- `--explore-fraction` — Random exploration rate for training data collection (default: 0.0)

Output:
- `recipes.json` — Extracted recipe data
- `crawl_logs/crawl_*.jsonl` — Event log for replay analysis
- `results/run_*.json` — Performance report

### Replay & Analyze

```bash
python -m cookie_monster_crawl.replay crawl_logs/crawl_*.jsonl
python -m cookie_monster_crawl.replay crawl_logs/crawl_*.jsonl --model models/model_v34.pkl --show-misses
```

### Train a Model

```bash
python -m cookie_monster_crawl.train crawl_logs/crawl_*.jsonl --model logistic_regression
```

### Strategy Generation

```bash
python -m cookie_monster_crawl.strategy replay_output.json --model models/model_v34.pkl
```

### Full Pipeline

```bash
python -m cookie_monster_crawl.pipeline --model models/model_v34.pkl --seeds data/static-target.json
```

Chains crawl, replay, strategy, apply, and train into a single run.

## Project Structure

```
cookie_monster_crawl/
├── crawler.py          # Async crawler with priority queue, domain locks, batch rescore
├── parser.py           # Link extraction, URL canonicalization, JSON-LD + microdata parsing
├── utils.py            # URLPrioritizer (ML + fallback scoring), RobotsChecker
├── train.py            # Multi-model training pipeline
├── replay.py           # Crawl log reconstruction and offline analysis
├── strategy.py         # Multi-agent strategy: Propose → Critique → Revise
├── outcomes.py         # Strategy outcome tracking for Critic history
├── apply.py            # Apply strategy changes to seed/segment files
├── investigation.py    # Tool framework for agent investigations
├── pipeline.py         # End-to-end pipeline orchestration
├── crawl_logger.py     # JSONL event logger
├── priority_queue.py   # Min-heap with random tie-breaking
data/
├── static-target.json              # 52 seed URLs
├── infrastructure_segments.txt     # Non-recipe URL keywords
├── navigational_segments.txt       # Index/listing URL keywords
├── recipe_related_segments.txt     # Recipe content URL keywords
└── crawl_config.json               # Scoring hyperparameters
tests/
├── test_parser.py
├── test_url_prioritizer.py
├── test_crawler.py
├── test_crawl_logger.py
└── test_crawler_log_integration.py
```

## Tests

```bash
python -m pytest tests/ -q
```
