"""LLM agents: each maps (system prompt, user prompt) to a text reply.

An agent may also set `last_info` (a small dict, e.g. finish reason and token counts)
after each call; the runner logs it next to the reply.
"""

import json
import subprocess
import urllib.request


class ClaudeCLIAgent:
    """Claude through the Claude Code CLI in print mode, with every tool disabled.

    Uses the CLI's own login, so no API key is needed. Each call cold-starts the
    CLI, which adds a few seconds per reply.
    """

    def __init__(self, model: str = "haiku", command: str = "claude", timeout: float = 300):
        self.model = model
        self.command = command
        self.timeout = timeout
        self.name = f"claude-{model}"

    def __call__(self, system: str, prompt: str) -> str:
        cmd = [
            self.command, "-p",
            "--model", self.model,
            "--system-prompt", system,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format", "json",
        ]
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=self.timeout, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.command} exited {proc.returncode}: "
                               f"{(proc.stderr or proc.stdout).strip()[:500]}")
        reply = json.loads(proc.stdout)
        if reply.get("is_error"):
            raise RuntimeError(f"{self.command} returned an error: {reply.get('result')}")
        return reply["result"]


class OpenAIChatAgent:
    """Any OpenAI-compatible chat endpoint, such as a vLLM server running an open model.

    With a reasoning model the reply is "<think>reasoning</think>" followed by the
    answer, so the log shows both; the runner strips the thinking before the
    environment or later turns see it.
    """

    def __init__(self, base_url: str, model: str, name: str | None = None,
                 max_tokens: int = 8192, temperature: float = 0.7, think: bool = True,
                 timeout: float = 900):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.think = think
        self.timeout = timeout
        self.name = name or (model if think else f"{model}-nothink")
        self.last_info = None

    def __call__(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": self.think},
        }
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.load(response)
        choice = reply["choices"][0]
        message = choice["message"]
        usage = reply.get("usage") or {}
        self.last_info = {"finish_reason": choice.get("finish_reason"),
                          "completion_tokens": usage.get("completion_tokens")}
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        return with_thinking(reasoning, message.get("content") or "")


class BedrockAgent:
    """An open model hosted on Amazon Bedrock, through the Converse API.

    Uses the default AWS credentials (`aws login`). Some models reject a
    system prompt; for those it is sent at the top of the user message instead.
    """

    def __init__(self, model_id: str, region: str = "us-east-1", max_tokens: int = 8192,
                 temperature: float = 0.7, client=None):
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=region)
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = model_id
        self.system_in_prompt = False
        self.last_info = None

    def __call__(self, system: str, prompt: str) -> str:
        try:
            reply = self._converse(system, prompt)
        except Exception as exc:
            if self.system_in_prompt or "system" not in str(exc).lower():
                raise
            self.system_in_prompt = True
            reply = self._converse(system, prompt)
        blocks = reply["output"]["message"]["content"]
        text = "".join(b.get("text", "") for b in blocks)
        reasoning = "".join(b["reasoningContent"]["reasoningText"].get("text", "")
                            for b in blocks if "reasoningContent" in b)
        stop = reply.get("stopReason")
        self.last_info = {"finish_reason": "length" if stop == "max_tokens" else stop,
                          "completion_tokens": reply.get("usage", {}).get("outputTokens")}
        return with_thinking(reasoning, text)

    def _converse(self, system: str, prompt: str) -> dict:
        kwargs = {}
        if self.system_in_prompt:
            prompt = f"{system}\n\n{prompt}"
        else:
            kwargs["system"] = [{"text": system}]
        return self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
            **kwargs,
        )


def with_thinking(reasoning: str | None, text: str) -> str:
    """Join a reasoning model's thinking and answer the way the runner and viewer expect."""
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{text}" if reasoning else text
