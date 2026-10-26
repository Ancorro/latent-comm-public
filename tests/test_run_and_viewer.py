"""Runner, agent and viewer tests with fake agents and a local stub server; no real
model calls."""

import json
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gradproposal.agents import BedrockAgent, OpenAIChatAgent
from gradproposal.run import play_episode, render_prompt, strip_thinking
from gradproposal.viewer import make_handler, read_episodes


class BinarySearchAgent:
    """Plays GuessTheNumber by bisecting on the higher/lower feedback in its prompt."""

    name = "bisect"

    def __call__(self, system, prompt):
        lo, hi = 1, 20
        for guess, direction in re.findall(r"guessed (\d+), and the target number is (\w+)",
                                           prompt):
            if direction == "higher":
                lo = max(lo, int(guess) + 1)
            else:
                hi = min(hi, int(guess) - 1)
        guess = (lo + hi) // 2
        return f"<think>between {lo} and {hi}</think>\nI'll try \\boxed{{{guess}}}"


class ConstantAgent:
    name = "constant"

    def __init__(self, reply):
        self.reply = reply

    def __call__(self, system, prompt):
        return self.reply


class FailingAgent:
    name = "failing"

    def __call__(self, system, prompt):
        raise RuntimeError("boom")


def test_multi_turn_episode_wins_and_logs_each_turn(tmp_path):
    log = tmp_path / "episodes.jsonl"
    record = play_episode(BinarySearchAgent(), "game:GuessTheNumber-v0", 0, 10, "r", log)

    assert record["status"] == "done" and 1 <= record["turns"] <= 5
    assert record["return"] > 0
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["env"] + ["agent", "env"] * record["turns"]
    # Each agent turn saw every earlier message, minus earlier thinking.
    last_agent = [m for m in record["messages"] if m["role"] == "agent"][-1]
    assert last_agent["prompt"] == render_prompt(record["messages"][:-2])
    if record["turns"] > 1:
        assert "<think>" not in last_agent["prompt"]
    snapshots = [json.loads(line) for line in log.read_text().splitlines()]
    assert {s["id"] for s in snapshots} == {record["id"]} and len(snapshots) >= 2
    assert read_episodes(tmp_path) == [record]


def test_single_turn_reasoning_gym_episode(tmp_path):
    record = play_episode(ConstantAgent("\\boxed{-1}"), "rg:leg_counting", 0, 10, "r",
                          tmp_path / "e.jsonl")
    assert record["turns"] == 1 and record["status"] == "done" and record["return"] == 0.0


def test_turn_limit_and_agent_error(tmp_path):
    log = tmp_path / "e.jsonl"
    stuck = play_episode(ConstantAgent("\\boxed{1}"), "game:GuessTheNumber-v0", 0, 2, "r", log)
    assert stuck["status"] in ("turn limit", "done") and stuck["turns"] <= 2
    failed = play_episode(FailingAgent(), "game:GuessTheNumber-v0", 0, 5, "r", log)
    assert failed["status"] == "error" and failed["error"] == "boom" and failed["turns"] == 0


def test_thinking_is_hidden_from_env_and_later_turns():
    reply = "<think>\nmaybe 7?\n</think>\n\nGuess: \\boxed{7}"
    assert strip_thinking(reply) == "Guess: \\boxed{7}"
    messages = [{"role": "env", "content": "start"}, {"role": "agent", "content": reply},
                {"role": "env", "content": "lower"}]
    assert "maybe" not in render_prompt(messages)


def test_openai_agent_against_stub_server():
    seen = {}

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({
                "choices": [{"finish_reason": "stop", "message": {
                    "content": "\\boxed{3}", "reasoning_content": " hmm "}}],
                "usage": {"completion_tokens": 12}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = OpenAIChatAgent(f"http://127.0.0.1:{server.server_address[1]}/v1", "m",
                                think=False)
        reply = agent("sys", "go")
    finally:
        server.shutdown()
    assert reply == "<think>\nhmm\n</think>\n\n\\boxed{3}"
    assert agent.name == "m-nothink"
    assert agent.last_info == {"finish_reason": "stop", "completion_tokens": 12}
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


class FakeBedrock:
    """Stands in for a boto3 bedrock-runtime client; rejects system prompts if told to."""

    def __init__(self, reject_system=False):
        self.reject_system = reject_system
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_system and "system" in kwargs:
            raise RuntimeError("ValidationException: This model doesn't support system messages")
        return {"output": {"message": {"content": [
                    {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
                    {"text": "\\boxed{4}"}]}},
                "stopReason": "max_tokens", "usage": {"outputTokens": 9}}


def test_bedrock_agent_parses_reasoning_and_stop_reason():
    client = FakeBedrock()
    agent = BedrockAgent("google.gemma-3-4b-it", client=client)
    assert agent("sys", "go") == "<think>\nhmm\n</think>\n\n\\boxed{4}"
    assert agent.name == "google.gemma-3-4b-it"
    assert agent.last_info == {"finish_reason": "length", "completion_tokens": 9}
    assert client.calls[0]["system"] == [{"text": "sys"}]


def test_bedrock_agent_moves_rejected_system_prompt_into_user_message():
    client = FakeBedrock(reject_system=True)
    agent = BedrockAgent("mistral.ministral-3-8b-instruct", client=client)
    agent("sys", "go")
    agent("sys", "again")
    assert agent.system_in_prompt
    assert len(client.calls) == 3  # one rejected call, then the system prompt rides along
    assert client.calls[-1]["messages"][0]["content"][0]["text"] == "sys\n\nagain"
    assert "system" not in client.calls[-1]


def test_read_episodes_merges_logs_keeps_latest_snapshot_and_skips_partial_line(tmp_path):
    (tmp_path / "local.jsonl").write_text(
        '{"id": "a", "time": "2", "v": 1}\n{"id": "a", "time": "2", "v": 2}\n{"id": "c',
        encoding="utf-8")
    (tmp_path / "hpc-1.jsonl").write_text(
        '{"id": "b", "time": "1"}\n{"id": "b", "time": "1"}\n', encoding="utf-8")
    assert read_episodes(tmp_path) == [{"id": "b", "time": "1"},
                                       {"id": "a", "time": "2", "v": 2}]
    assert read_episodes(tmp_path / "missing") == []


def test_viewer_serves_page_and_episodes(tmp_path):
    (tmp_path / "episodes.jsonl").write_text('{"id": "a", "return": 1.0}\n', encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        lesson = urllib.request.urlopen(base + "/lesson").read().decode()
        visuals = urllib.request.urlopen(base + "/visuals").read().decode()
        episodes = json.loads(urllib.request.urlopen(base + "/api/episodes").read())
    finally:
        server.shutdown()
    assert "Agent I/O" in page and "api/episodes" in page and 'href="lesson"' in page
    assert "Reinforcement learning" in lesson and "api/episodes" in lesson
    assert 'href="visuals"' in lesson and "youtube-nocookie.com/embed/" in visuals
    assert episodes == [{"id": "a", "return": 1.0}]
