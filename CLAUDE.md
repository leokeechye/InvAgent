# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for the paper *InvAgent: A Large Language Model based Multi-Agent System for Inventory Management in Supply Chains* (arxiv.org/abs/2407.11384). It contains:

- A multi-echelon inventory-management gym environment (Ray RLlib `MultiAgentEnv`).
- A heuristic baseline policy and two PPO variants (IPPO with parameter sharing, MAPPO with a centralized critic) trained via Ray Tune.
- An AutoGen notebook where each supply-chain stage is driven by a GPT-4 `ConversableAgent`.

There is no test suite, no linter config, and no build step. All entry points are `python` scripts/notebooks.

## Setup and execution

```bash
pip install -r requirements.txt
```

Versions are pinned and *not* arbitrary — `gym==0.26.2` and `gymnasium==0.28.1` coexist because Ray RLlib 2.24 needs both; `pyautogen==0.2.28` is the pre-AG2 split. Don't bump these without a reason.

**All scripts must be run from `src/`**, not from the repo root:

```bash
cd src
python env.py        # smoke test the environment
python baseline.py   # evaluate fixed heuristic on all env configs (100 episodes each)
python ippo.py       # Ray Tune sweep, 20 trials, IPPO + shared policy
python mappo.py      # Ray Tune sweep, 20 trials, MAPPO + centralized critic
```

Two reasons running from `src/` matters:
1. The `src/` files use bare imports (`from config import ...`, `from env import ...`) — there is no package `__init__.py`.
2. `ippo.py` and `mappo.py` write Ray Tune results to `os.path.join(os.getcwd(), "..", "results")`, i.e. they assume CWD is `src/` and produce a sibling `results/` directory.

For the notebook (`notebooks/autogen.ipynb`):
- Requires `OPENAI_API_KEY` in the environment (uses `gpt-4` by default).
- Imports `from env` / `from config`, so the Jupyter kernel needs `src/` on `sys.path` (the notebook does not add it itself — launch Jupyter from `src/`, or prepend `sys.path.insert(0, '../src')`).

## Architecture

### Environment (`src/env.py`)

`InventoryManagementEnv` is a serial multi-echelon supply chain: stage `0` is the retailer (sees customer demand), stage `M-1` is the manufacturer (orders materialize as production after its lead time). Each stage's downstream orders become its sales; its upstream is the next stage. Time advances in discrete periods; the per-period sequence is **deliver → order/demand → fulfill → bookkeep profit** (see the docstring).

The per-stage observation is a fixed-length `MultiDiscrete` vector built in `update_state()`:

```
[prod_capacity, sale_price, order_cost, backlog_cost, holding_cost,
 lead_time, inventory_{t-1}, backlog_{t-1}, upstream_backlog_{t-1},
 sales over the last max_lead_time periods (zero-padded),
 arriving orders over the last lead_time periods]
```

`_parse_state` / `parse_state` decode this back to a dict — useful when writing policies that read the observation semantically (the baseline and the LLM agents both do this). `agent_observation_space` / `agent_action_space` are the per-stage spaces; the env-level `observation_space` / `action_space` are dicts keyed by `stage_{m}`.

Scenarios live in `env_configs` in `src/config.py` (`two_agent`, `constant_demand`, `variable_demand`, `larger_demand`, `seasonal_demand`, `normal_demand`). `env_creator(env_config)` is the factory registered with Ray as `"InventoryManagementEnv"`.

### Policies

- **`src/baseline.py`** — `FixedPolicy` is a Ray RLlib `Policy` subclass (not a learned model). `get_fixed_inventory_action` implements an order-up-to rule with two modes: `'production'` targets `prod_capacity`, `'sale'` targets `mean(recent_sales) * lead_time + backlog`. Note `compute_actions` first calls `decode_obs` to invert RLlib's one-hot encoding of the `MultiDiscrete` observation before parsing — keep that step if you add new fixed policies.
- **`src/ippo.py`** — One shared policy across all stages (parameter sharing). `policy_mapping_fn` maps every agent id to `"shared_policy"`.
- **`src/mappo.py`** — Per-stage policies (`policy_0`, `policy_1`, ...) each using the custom `CentralizedCriticModel` (a `TorchModelV2` with a shared trunk and a separate value head). Registered with `ModelCatalog.register_custom_model("centralized_critic", ...)`.

Both PPO scripts hardcode `env_config_name = "constant_demand"` inside `tune_ppo`; change that string to sweep a different scenario. The Tune search space (`lr`, `train_batch_size`, `sgd_minibatch_size`, `num_sgd_iter`, `training_iteration`, `fcnet_hiddens`) is also defined inline at the bottom of each file.

### LLM agents (`notebooks/autogen.ipynb`)

Wraps each stage in an AutoGen `ConversableAgent`, prompts it with a natural-language rendering of the parsed state (`get_state_description`) plus a demand description (`get_demand_description` — must be extended if you add a new env config), and parses the integer order back out of the LLM reply. The notebook is the only place the LLM-multi-agent system actually runs; nothing in `src/` calls AutoGen.

### Streamlit GUI + Railway deployment

`app.py` (repo root) is a Streamlit port of `notebooks/autogen.ipynb` — same prompt, same `[N]` regex parsing, same per-round step loop, but driven by a sidebar (scenario, model, seed, optional API-key override) with per-round expanders, progress bar, and a final per-stage reward bar chart. It prepends `src/` to `sys.path` and imports only `env` + `config` (never `baseline`/`ippo`/`mappo`, which pull in Ray-RLlib trainer machinery that isn't needed to serve the AutoGen simulation).

Deployment to Railway uses the same recipe as `~/TradingAgents`:
- `Dockerfile` — `python:3.11-slim` + `gcc g++ git` system deps + `pip install -r requirements.txt`, runs `streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true`.
- `railway.toml` — `DOCKERFILE` builder, 300s healthcheck timeout, `ON_FAILURE` restart × 3.
- `.env.example` — only `OPENAI_API_KEY=` (placeholder). Real key must live in Railway's Variables tab, never in a committed file.
- `.dockerignore` — excludes `notebooks/`, `results/`, `.env`, `.venv`, `.claude/`, etc.

`requirements.txt` adds `streamlit>=1.32,<2` and `python-dotenv>=1.0,<2` on top of the pinned research deps. The resulting image is large (~3 GB) because `torch` and `ray` are still required — `env.py` subclasses `ray.rllib.env.multi_agent_env.MultiAgentEnv` so Ray must be importable even though no trainer is launched.
