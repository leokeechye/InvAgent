"""Streamlit GUI for the InvAgent multi-agent LLM supply-chain simulation."""
import os
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from env import env_creator  # noqa: E402
from config import env_configs  # noqa: E402
from autogen import ConversableAgent  # noqa: E402

load_dotenv()

st.set_page_config(page_title="InvAgent — LLM Supply Chain", page_icon="📦", layout="wide")
st.title("InvAgent — LLM Multi-Agent Inventory Management")
st.caption(
    "LLM agents (one per supply-chain stage) place orders each round to minimize "
    "total cost. Based on arxiv.org/abs/2407.11384."
)


DEMAND_DESCRIPTIONS = {
    "two_agent": "The expected demand at the retailer is a constant 4 units per round.",
    "constant_demand": "The expected demand at the retailer (stage 1) is a constant 4 units for all 12 rounds.",
    "variable_demand": "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 4} for all 12 rounds.",
    "larger_demand": "The expected demand at the retailer (stage 1) is a discrete uniform distribution U{0, 8} for all 12 rounds.",
    "seasonal_demand": "The expected demand at the retailer (stage 1) is U{0, 4} for the first 4 rounds and U{5, 8} for the last 8 rounds.",
    "normal_demand": "The expected demand at the retailer (stage 1) is N(4, 2^2), truncated at 0, for all 12 rounds.",
}


def get_state_description(state):
    deliveries_window = state["deliveries"][-state["lead_time"]:] if state["lead_time"] > 0 else []
    return (
        f" - Lead Time: {state['lead_time']} round(s)\n"
        f" - Inventory Level: {state['inventory']} unit(s)\n"
        f" - Current Backlog (you owing to the downstream): {state['backlog']} unit(s)\n"
        f" - Upstream Backlog (your upstream owing to you): {state['upstream_backlog']} unit(s)\n"
        f" - Previous Sales (recent rounds, old to new): {state['sales']}\n"
        f" - Arriving Deliveries (this and next rounds, near to far): {deliveries_window}"
    )


def build_prompt(period, stage, num_stages, stage_state, demand_description, downstream_order):
    return (
        f"Now this is the round {period + 1}, "
        f"and you are at the stage {stage + 1} of {num_stages} in the supply chain. "
        f"Given your current state:\n{get_state_description(stage_state)}\n\n"
        f"{demand_description} {downstream_order}"
        "What is your action (order quantity) for this round?\n\n"
        'Golden rule of this game: Open orders should always equal to "expected downstream orders + backlog". '
        "If open orders are larger than this, the inventory will rise (once the open orders arrive). "
        "If open orders are smaller than this, the backlog will not go down and it may even rise. "
        "Please consider the lead time and place your order in advance. "
        "Remember that your upstream has its own lead time, so do not wait until your inventory runs out. "
        "Also, avoid ordering too many units at once. "
        "Try to spread your orders over multiple rounds to prevent the bullwhip effect. "
        "Anticipate future demand changes and adjust your orders accordingly to maintain a stable inventory level.\n\n"
        "Please state your reason in 1-2 sentences first "
        "and then provide your action as a non-negative integer within brackets (e.g. [0])."
    )


with st.sidebar:
    st.header("Configuration")
    scenario_names = list(env_configs.keys())
    default_idx = scenario_names.index("seasonal_demand") if "seasonal_demand" in scenario_names else 0
    env_config_name = st.selectbox("Scenario", scenario_names, index=default_idx)
    model = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4 is what the paper used. gpt-4o-mini is much cheaper.",
    )
    api_key_override = st.text_input(
        "OPENAI_API_KEY (optional override)",
        value="",
        type="password",
        help="Leave empty to use the OPENAI_API_KEY environment variable set on Railway.",
    )
    seed = st.number_input("NumPy seed", min_value=0, value=42, step=1)
    run_button = st.button("Run simulation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Selected scenario**")
    cfg = env_configs[env_config_name]
    st.json(
        {
            "num_stages": cfg["num_stages"],
            "num_periods": cfg["num_periods"],
            "stage_names": cfg["stage_names"],
            "lead_times": cfg["lead_times"],
            "prod_capacities": cfg["prod_capacities"],
        },
        expanded=False,
    )

if not run_button:
    st.info("Pick a scenario and model in the sidebar, then click **Run simulation**.")
    st.stop()

effective_key = (api_key_override.strip() or os.environ.get("OPENAI_API_KEY", "")).strip()
if not effective_key:
    st.error(
        "No OPENAI_API_KEY available. Set it in the sidebar, or configure it as an "
        "environment variable in Railway's Variables tab."
    )
    st.stop()

np.random.seed(int(seed))
env_config = env_configs[env_config_name]
im_env = env_creator(env_config)
im_env.reset()

llm_config = {"model": model, "api_key": effective_key}
user_proxy = ConversableAgent(name="User Proxy", llm_config=False, human_input_mode="NEVER")
stage_agents = [
    ConversableAgent(
        name=f"{name.capitalize()} Agent",
        system_message=(
            f"You play a crucial role in a {len(env_config['stage_names'])}-stage supply chain "
            f"as the stage {i + 1} ({name}). Your goal is to minimize the total cost by managing "
            "inventory and orders effectively."
        ),
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    for i, name in enumerate(env_config["stage_names"])
]

demand_description = DEMAND_DESCRIPTIONS.get(env_config_name, "")
progress = st.progress(0.0, text="Starting simulation…")
log_container = st.container()
total_reward = 0
total_cost = 0.0
per_stage_rewards = {f"stage_{m}": 0 for m in range(im_env.num_stages)}

for period in range(im_env.num_periods):
    state_dict = im_env.parse_state(im_env.state_dict)
    action_dict: dict[str, int] = {}

    with log_container.expander(
        f"Round {period + 1} of {im_env.num_periods}", expanded=(period == 0)
    ):
        for stage in range(im_env.num_stages):
            stage_state = state_dict[f"stage_{stage}"]
            downstream_order = (
                f"Your downstream order from the stage {stage} for this round is "
                f"{action_dict[f'stage_{stage - 1}']}. "
                if stage != 0
                else ""
            )
            message = build_prompt(
                period, stage, im_env.num_stages, stage_state, demand_description, downstream_order
            )

            with st.spinner(
                f"Stage {stage + 1} ({env_config['stage_names'][stage]}) thinking…"
            ):
                chat_result = user_proxy.initiate_chat(
                    stage_agents[stage],
                    message=message,
                    summary_method="last_msg",
                    max_turns=1,
                    clear_history=False,
                )

            chat_summary = chat_result.summary or ""
            try:
                total_cost += chat_result.cost["usage_including_cached_inference"]["total_cost"]
            except (KeyError, TypeError, AttributeError):
                pass

            match = re.search(r"\[(\d+)\]", chat_summary)
            action = int(match.group(1)) if match else 0
            action_dict[f"stage_{stage}"] = action

            st.markdown(
                f"**Stage {stage + 1} — {env_config['stage_names'][stage].capitalize()}** "
                f"→ order: `{action}`"
            )
            st.text(chat_summary)

        next_states, rewards, *_ = im_env.step(action_dict)
        for k, v in rewards.items():
            per_stage_rewards[k] += int(v)
        total_reward += int(sum(rewards.values()))
        st.write(f"Rewards this round: {rewards}")
        st.write(
            f"Running total reward: **{total_reward}** · "
            f"API cost so far: **${total_cost:.4f}**"
        )

    progress.progress(
        (period + 1) / im_env.num_periods,
        text=f"Round {period + 1} of {im_env.num_periods}",
    )

progress.empty()
st.success(
    f"Simulation complete · total reward: **{total_reward}** · API cost: **${total_cost:.4f}**"
)
st.subheader("Per-stage cumulative rewards")
st.bar_chart(per_stage_rewards)
