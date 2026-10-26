"""Web page showing agent inputs and outputs from episode logs, refreshed live,
plus an interactive reinforcement-learning lesson at /lesson built on those episodes
and a page of live RL demos and game videos at /visuals.

Reads every runs/*.jsonl: local runs plus logs streamed back from HPC.

Example:
    python -m gradproposal.viewer --port 8790
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("viewer.html")
LESSON = Path(__file__).with_name("lesson.html")
VISUALS = Path(__file__).with_name("visuals.html")


def read_episodes(runs_dir: Path) -> list[dict]:
    """Latest snapshot of each episode across all logs, oldest episode first.

    The runner appends a snapshot per turn under the same id; a half-written last
    line is skipped.
    """
    latest = {}
    for log_path in sorted(runs_dir.glob("*.jsonl")):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[record["id"]] = record
    return sorted(latest.values(), key=lambda r: r.get("time", ""))


def make_handler(runs_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            elif path == "/lesson":
                self._send(200, "text/html; charset=utf-8", LESSON.read_bytes())
            elif path == "/visuals":
                self._send(200, "text/html; charset=utf-8", VISUALS.read_bytes())
            elif path == "/api/episodes":
                body = json.dumps(read_episodes(runs_dir)).encode()
                self._send(200, "application/json", body)
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, status: int, content_type: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # keep the console quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.runs_dir))
    print(f"serving {args.runs_dir}/*.jsonl on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
