"""Benchmarks, prompts and scoring for the LatentMAS reproduction.

Prompts are copied verbatim from the official code (github.com/Gen-Verse/LatentMAS, prompts.py at
commit 9a9e4d3), sequential setting only, typos included. Answer parsing copies its utils.py.

Build the fixed subsets (seeded, committed) with:
    python -m gradproposal.latent.bench --n 300 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "latent" / "data"
_SP = " "  # official prompt has a trailing space here; editors strip literal ones
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
ROLES = ("planner", "critic", "refiner", "judger")
BENCHES = ("gsm8k", "arc_challenge")

# ---------------------------------------------------------------- data


def _gsm8k_gold(solution: str) -> str | None:
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", solution)  # official extract_gold
    return m.group(1).strip().lower() if m else None


def _arc_question(item) -> tuple[str, str]:
    label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

    def lab(s):
        return label_map.get(str(s).strip(), str(s).strip().lower())

    ch = item["choices"]
    lines = [f"{lab(k)}: {t.strip()}" for k, t in zip(ch["label"], ch["text"], strict=True)]
    return item["question"].strip() + "\n" + "\n".join(lines), lab(item["answerKey"])


def build_subsets(n: int, seed: int) -> None:
    from datasets import load_dataset

    DATA.mkdir(parents=True, exist_ok=True)
    sources = {
        "gsm8k": load_dataset("openai/gsm8k", "main", split="test"),
        "arc_challenge": load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"),
    }
    for bench, ds in sources.items():
        idx = sorted(random.Random(seed).sample(range(len(ds)), n))
        with open(DATA / f"{bench}_{n}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for i in idx:
                item = ds[i]
                if bench == "gsm8k":
                    q, gold = item["question"].strip(), _gsm8k_gold(item["answer"])
                else:
                    q, gold = _arc_question(item)
                row = {"id": f"{bench}-test-{i}", "question": q, "gold": gold}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{bench}: {n} of {len(ds)} -> {DATA / f'{bench}_{n}.jsonl'}")


def load(bench: str, n: int = 300, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(DATA / f"{bench}_{n}.jsonl", encoding="utf-8")]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- scoring (official utils.py)


def extract_answer(text: str) -> str | None:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        number = re.search(r"[-+]?\d+(?:\.\d+)?", boxes[-1])
        ans = number.group(0) if number else boxes[-1].strip()
    else:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        ans = numbers[-1] if numbers else None
    return ans.strip().lower() if ans is not None else None


def score(text: str, gold: str) -> tuple[str | None, bool]:
    pred = extract_answer(text)
    return pred, bool(pred and gold and pred == gold)


# ---------------------------------------------------------------- prompts


def _messages(user: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_ANSWER = {
    "gsm8k": "",
    "arc_challenge": "Your final answer must be selected from A,B,C,D. For example \\boxed{A}. "
    "Do not add any other contents inside the box.\n",
}


def latent_messages(role: str, question: str, bench: str) -> list[dict]:
    """build_agent_message_sequential_latent_mas."""
    if role == "planner":
        user = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""  # noqa: E501
    elif role == "critic":
        user = f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""  # noqa: E501
    elif role == "refiner":
        user = f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""  # noqa: E501
    else:
        user = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.{_SP}

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
{_ANSWER[bench]}
Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""  # noqa: E501
    return _messages(user)


def textmas_messages(role: str, question: str, context: str, bench: str) -> list[dict]:
    """build_agent_messages_sequential_text_mas (context = earlier agents' outputs, untruncated)."""
    if role == "planner":
        user = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""  # noqa: E501
    elif role == "critic":
        user = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{context}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""  # noqa: E501
    elif role == "refiner":
        user = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{context}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""  # noqa: E501
    else:
        user = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{context}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
{_ANSWER[bench]}
Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""  # noqa: E501
    return _messages(user)


def single_messages(question: str, bench: str) -> list[dict]:
    """build_agent_messages_single_agent."""
    user = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
{_ANSWER[bench]}
Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""  # noqa: E501
    return _messages(user)


# ---------------------------------------------------------------- FanOutQA (CacheBack)
#
# The CacheBack panel (github.com/maxr0ssi/rclc, tag paper-v1, Apache-2.0): 50 FanOutQA dev
# questions, the first qualifying ids by sha256 of the question id, 3 workers each holding 40K-50K
# Qwen tokens of cleaned Wikipedia text. Copied unchanged from the bundle's items/*.json.

FANOUTQA = DATA / "fanoutqa_dev50.jsonl.gz"
FANOUT_WORKERS = 3


def build_fanoutqa(bundle: Path) -> None:
    """Copy the 50 selected items (question, raw gold answer, worker texts, page provenance)."""
    import gzip
    import hashlib

    panel = json.loads((bundle / "panel.json").read_text(encoding="utf-8"))
    dev = {q["id"]: q for q in json.loads(
        (bundle / "source_cache" / "fanout-final-dev.json").read_text(encoding="utf-8"))}
    picked = sorted((i for i in panel["items"] if i["role"] == "selected"), key=lambda i: i["rank"])
    assert len(picked) == 50, len(picked)
    with gzip.open(FANOUTQA, "wt", encoding="utf-8", newline="\n") as f:
        for summary in picked:
            item = json.loads((bundle / "items" / f"{summary['qid']}.json").read_text("utf-8"))
            raw = dev[item["qid"]]
            assert raw["question"] == item["question"]
            workers = []
            for w in item["workers"]:
                assert hashlib.sha256(w["text"].encode()).hexdigest() == w["text_sha256"]
                workers.append({
                    "text": w["text"],
                    "pages": [{"pageid": p["pageid"], "title": p["title"]} for p in w["pages"]],
                    "qwen_prompt_tokens": w["rendered_worker_prompt_tokens"]["qwen"],
                })
            row = {"id": item["qid"], "question": item["question"], "answer": raw["answer"],
                   "categories": item["categories"], "workers": workers}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"fanoutqa: {len(picked)} items -> {FANOUTQA}")


def load_fanoutqa(limit: int | None = None) -> list[dict]:
    import gzip

    with gzip.open(FANOUTQA, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


# Official prompt bytes (rcc/benchmarks/fanoutqa/prompts.py, "natural" style). The line breaks
# inside the worker brief are part of the prompt: the panel was tokenized with them.
_WORKER_BRIEF = """You are one reasoning worker. Inspect only your
private evidence. Work through every relevant name, number, relationship, and
uncertainty that could help the coordinator answer the question. Return a complete
reasoning handoff, including intermediate reasoning and unresolved alternatives.
Do not claim access to any other worker's evidence."""
_WORKER_CLOSE = "Return the complete reasoning handoff now."


def worker_body(question: str, evidence: str) -> str:
    return (f"{_WORKER_BRIEF}\n\nPrivate evidence:\n{evidence}"
            f"\n\nQuestion: {question}\n\n{_WORKER_CLOSE}")


def report_payload(reports: list[str]) -> str:
    """Three visible reports, as the official Qwen coordinator receives them."""
    return "\n\n".join(f"[worker {i} report]\n{r.strip()}" for i, r in enumerate(reports))


def manager_body(question: str, reports: str = "") -> str:
    """Coordinator (receiver) prompt; `reports` empty for the latent and question-only arms."""
    block = f"\n\nWorker reports:\n{reports}" if reports else ""
    return (
        "You are the coordinator. Answer from the supplied handoff and do not "
        f"invent missing evidence.{block}\n\nQuestion: {question}\n\n"
        "Give the answer directly and include every requested leaf."
    )


def render(tokenizer, body: str) -> str:
    """Chat template with Qwen3 thinking on (the model opens <think> itself)."""
    return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)


def question_span(tokenizer, rendered: str, question: str) -> tuple[list[int], int, int]:
    """Token ids of a rendered prompt and the [lo, hi) rows covering the question text (the
    official `question_token_ids` offset rule: tokens overlapping the last occurrence)."""
    start = rendered.rfind(question)
    if start < 0:
        raise ValueError("question not found in the rendered prompt")
    end = start + len(question)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    rows = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end]
    return list(enc["input_ids"]), rows[0], rows[-1] + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fanoutqa-bundle", type=Path, default=None,
                    help="unpacked rclc paper-v1 data/fanoutqa-m3-source-*.tar.zst")
    a = ap.parse_args()
    if a.fanoutqa_bundle:
        build_fanoutqa(a.fanoutqa_bundle)
    else:
        build_subsets(a.n, a.seed)
