"""Play an agent through GEM environments and log every exchange to a JSONL file.

GEM (https://github.com/axon-rl/gem) gives one reset/step interface for multi-turn
games (game:Wordle-v0-easy, game:GuessTheNumber-v0, ...) and single-turn Reasoning
Gym tasks (rg:leg_counting, ...).

Example:
    python -m gradproposal.run --envs game:GuessTheNumber-v0,rg:leg_counting --n 2
"""

import argparse
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import gem

from gradproposal.agents import BedrockAgent, ClaudeCLIAgent, OpenAIChatAgent

DEFAULT_LOG = Path("runs/episodes.jsonl")
SYSTEM_PROMPT = (
    "You are an agent acting in an environment. Read the task and any feedback, think "
    "briefly, then put your final answer or next move inside \\boxed{...}. Only the last "
    "\\boxed{} in your reply counts."
)


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks, as Qwen-style chat templates do for past turns."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def render_prompt(messages: list[dict]) -> str:
    """The text the agent sees this turn: the whole history, oldest first, minus the
    agent's earlier thinking.

    This is the state hand-off between turns; experiments can swap it out.
    """
    if len(messages) == 1:
        return messages[0]["content"]
    labels = {"env": "ENVIRONMENT", "agent": "YOU"}
    return "\n\n".join(f"[{labels[m['role']]}]\n{strip_thinking(m['content'])}"
                       for m in messages)


def env_text(obs: str, info: dict) -> str:
    suffix = info.get("suffix")
    return f"{obs}\n{suffix}" if suffix else obs


def play_episode(agent, env_id: str, seed: int, max_turns: int, run_id: str,
                 log_path: Path) -> dict:
    """Play one episode; append a snapshot of the record after every turn.

    Records share an `id`, so readers keep the last snapshot per id.
    """
    env = gem.make(env_id)
    obs, info = env.reset(seed=seed)
    record = {
        "id": uuid.uuid4().hex, "run": run_id,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent": agent.name, "env": env_id, "seed": seed, "system": SYSTEM_PROMPT,
        "messages": [{"role": "env", "content": env_text(obs, info)}],
        "return": 0.0, "turns": 0, "status": "running", "error": None,
    }
    messages = record["messages"]

    def save():
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    save()
    for _ in range(max_turns):
        prompt = render_prompt(messages)
        start = time.monotonic()
        try:
            reply = agent(SYSTEM_PROMPT, prompt)
        except Exception as exc:  # log the failure and stop this episode
            record.update(status="error", error=str(exc))
            break
        messages.append({"role": "agent", "content": reply, "prompt": prompt,
                         "latency_s": round(time.monotonic() - start, 2),
                         "info": getattr(agent, "last_info", None)})
        obs, reward, terminated, truncated, info = env.step(strip_thinking(reply))
        record["turns"] += 1
        record["return"] += float(reward)
        messages.append({"role": "env", "content": env_text(obs, info), "reward": float(reward)})
        if terminated or truncated:
            record["status"] = "done"
            break
        save()
    else:
        record["status"] = "turn limit"
    save()
    return record


def run(agent, env_ids: list[str], n: int, seed: int, max_turns: int,
        log_path: Path) -> list[dict]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    records = []
    for env_id in env_ids:
        for i in range(n):
            r = play_episode(agent, env_id, seed + i, max_turns, run_id, log_path)
            records.append(r)
            print(f"{env_id} seed={seed + i} return={r['return']:.2f} turns={r['turns']} "
                  f"{r['status']}{': ' + r['error'] if r['error'] else ''}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", default="game:GuessTheNumber-v0",
                        help="comma-separated GEM environment ids")
    parser.add_argument("--n", type=int, default=1, help="episodes per environment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--agent", choices=["claude", "openai", "bedrock"], default="claude",
                        help="claude: Claude Code CLI; openai: OpenAI-compatible server; "
                             "bedrock: Amazon Bedrock model id")
    parser.add_argument("--model", default="haiku",
                        help="Claude model alias, served model name, or Bedrock model id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                        help="OpenAI-compatible endpoint (--agent openai)")
    parser.add_argument("--no-think", action="store_true",
                        help="turn off a reasoning model's thinking (--agent openai)")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    if args.agent == "claude":
        agent = ClaudeCLIAgent(args.model)
    elif args.agent == "bedrock":
        agent = BedrockAgent(args.model, max_tokens=args.max_tokens)
    else:
        agent = OpenAIChatAgent(args.base_url, args.model, max_tokens=args.max_tokens,
                                think=not args.no_think)
    records = run(agent, args.envs.split(","), args.n, args.seed, args.max_turns, args.log)
    mean = sum(r["return"] for r in records) / max(len(records), 1)
    print(f"{len(records)} episodes, mean return {mean:.2f}, log {args.log}")


if __name__ == "__main__":
    main()
