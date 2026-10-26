"""W&B tracking: the online-only guard, the run layout, summaries, the session wrapper (against a
fake wandb module), the registry and the backfill dry run. No network and no W&B account."""

import ast
import importlib.machinery
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
import types

import pytest

from gradproposal import tracking as T
from gradproposal.latent import cb_run, split
from gradproposal.latent import run as run_mod
from gradproposal.latent import wandb_backfill as wb

RUN_ARGS = {"bench": "gsm8k", "conditions": "latent,wrong,noise,nocache", "m": "40", "seeds": "0",
            "model": "Qwen/Qwen3-4B", "n": 300, "limit": None, "batch_size": 32,
            "partner_group": 32, "max_new_tokens": 2048, "temperature": 0.6, "top_p": 0.95,
            "think": True, "realign": True, "out": "latent/results/gsm8k-cache-s0.jsonl"}
SPLIT_ARGS = {"conditions": "latent,wrong,nocache,text", "n": 100, "m": 40, "seed": 0,
              "model": "Qwen/Qwen3-4B", "batch_size": 25, "max_new_tokens": 2048,
              "temperature": 0.6, "top_p": 0.95, "think": True,
              "out": "latent/results/split-gsm8k.jsonl", "report": None}
CB_ARGS = {"stage": "receive", "model": "Qwen/Qwen3-8B", "limit": None, "bank": "b",
           "reports": "r", "out_dir": "o", "arms": "question,cb4", "draws": "0,1,2", "seed": 0,
           "shard": "0/1", "max_new_tokens": 24000, "closer": 4096, "kv_budget": 300000}
LATENTMAS = T.RESULTS / "latentmas-qwen3-4b"
GOOD_ENV = {"WANDB_MODE": "online", "WANDB_API_KEY": "test-value"}

# ---------------------------------------------------------------- guard


def test_guard_accepts_online_cloud_with_credentials():
    assert T.wandb_env_errors(GOOD_ENV, netrc_ok=False) == []
    assert T.wandb_env_errors({}, netrc_ok=True) == []  # unset mode is pinned to online
    assert T.wandb_env_errors({**GOOD_ENV, "WANDB_BASE_URL": "https://api.wandb.ai/"}, False) == []
    assert T.wandb_env_errors({**GOOD_ENV, "WANDB_DISABLED": "false"}, False) == []


@pytest.mark.parametrize("mode", ["offline", "disabled", "dryrun", "shared", "", "ONLINE", "x"])
def test_guard_rejects_every_non_online_mode(mode):
    errors = T.wandb_env_errors({**GOOD_ENV, "WANDB_MODE": mode}, netrc_ok=True)
    assert len(errors) == 1 and "WANDB_MODE" in errors[0]


def test_guard_rejects_other_hosts_disabling_missing_credentials_and_package():
    assert "WANDB_BASE_URL" in T.wandb_env_errors(
        {**GOOD_ENV, "WANDB_BASE_URL": "http://localhost:8080"}, False)[0]
    assert "WANDB_DISABLED" in T.wandb_env_errors({**GOOD_ENV, "WANDB_DISABLED": "true"},
                                                  False)[0]
    assert "credentials" in T.wandb_env_errors({"WANDB_MODE": "online"}, netrc_ok=False)[0]
    assert "credentials" in T.wandb_env_errors({"WANDB_API_KEY": "  "}, netrc_ok=False)[0]
    assert "not installed" in T.wandb_env_errors(GOOD_ENV, False, wandb_installed=False)[0]


def test_require_wandb_pins_cloud_and_never_echoes_the_key(tmp_path):
    env = {"WANDB_API_KEY": "secret-value", "NETRC": str(tmp_path / "none")}
    T.require_wandb(env)
    assert env["WANDB_MODE"] == "online" and env["WANDB_BASE_URL"] == T.BASE_URL
    bad = {"WANDB_API_KEY": "secret-value", "WANDB_MODE": "offline"}
    with pytest.raises(T.TrackingError) as e:
        T.require_wandb(bad)
    assert "WANDB_MODE='offline'" in str(e.value) and "secret-value" not in str(e.value)
    assert bad["WANDB_MODE"] == "offline"  # nothing pinned on failure


def test_netrc_detection(tmp_path):
    p = tmp_path / "netrc"
    p.write_text("machine github.com login a password b\n"
                 "machine api.wandb.ai\n  login user\n  password abc123\n")
    assert T.netrc_has_wandb(p)
    p.write_text("machine api.wandb.ai login user\nmachine other password x\n")
    assert not T.netrc_has_wandb(p)
    p.write_text("machine example.com password x\n")
    assert not T.netrc_has_wandb(p)
    assert not T.netrc_has_wandb(tmp_path / "missing")
    assert T.netrc_path({"NETRC": str(p)}) == p


def _first_call(fn, names) -> int | None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and (
        getattr(n.func, "attr", None) in names or getattr(n.func, "id", None) in names)]
    return min(lines) if lines else None


@pytest.mark.parametrize("module", [run_mod, split, cb_run])
def test_entrypoints_check_wandb_before_loading_data_or_models(module):
    guard = _first_call(module.main, {"require_wandb"})
    folder = _first_call(module.main, {"require_results_folder"})
    load = _first_call(module.main, {"load", "load_split", "Runner", "SplitRunner", "Lab"})
    assert guard and folder and load and guard < folder < load
    assert "TrackedRun" in inspect.getsource(module.main)


def test_only_tracking_calls_wandb_directly():
    src = T.REPO / "src" / "gradproposal"
    offenders = [p.name for p in src.rglob("*.py") if p.name != "tracking.py"
                 and ("import wandb" in p.read_text(encoding="utf-8")
                      or "wandb.init(" in p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_offline_mode_aborts_runner_before_any_model_load(tmp_path):
    out = tmp_path / "x.jsonl"
    env = {**os.environ, "WANDB_MODE": "offline", "HF_HUB_OFFLINE": "1"}
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "gradproposal.latent.run", "--bench", "gsm8k",
                        "--model", "nonexistent/model", "--out", str(out)],
                       env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode != 0
    assert "refusing to start" in p.stderr and "WANDB_MODE='offline'" in p.stderr, p.stderr
    assert not out.exists() and time.perf_counter() - t0 < 120


# ---------------------------------------------------------------- layout


def test_layout_latentmas_runs():
    lay = T.layout_for("latentmas", RUN_ARGS, LATENTMAS / "gsm8k-cache-s0.jsonl",
                       git="47c4ee94d8a877d26349909c7944ea500fc741aa")
    assert (lay.project, lay.group, lay.job_type) == ("latent-comm", "latentmas-qwen3-4b", "eval")
    assert lay.name == "latentmas-qwen3-4b.gsm8k.latent+wrong+noise+nocache.m40.s0"
    assert lay.tags == ("git-47c4ee9", "qwen3-4b", "gsm8k", "paper-p1")
    assert lay.artifact == "results-latentmas-qwen3-4b-gsm8k"
    assert lay.out == "latent/results/latentmas-qwen3-4b/gsm8k-cache-s0.jsonl"
    T.require_results_folder(lay)
    single = T.layout_for("latentmas", {**RUN_ARGS, "conditions": "single", "seeds": "1"},
                          LATENTMAS / "gsm8k-single-s1.jsonl")
    assert single.name == "latentmas-qwen3-4b.gsm8k.single.s1"
    sweep = T.layout_for("latentmas", {**RUN_ARGS, "conditions": "latent", "m": "10,20"},
                         LATENTMAS / "gsm8k-sweep-s0.jsonl", backfill=True)
    assert sweep.name == "latentmas-qwen3-4b.gsm8k.latent.m10+20.s0"
    assert sweep.tags[-1] == "backfill"
    other = T.layout_for("latentmas", {**RUN_ARGS, "model": "Qwen/Qwen3.5-4B"},
                         T.RESULTS / "latentmas-qwen3.5-4b" / "x.jsonl")
    assert other.group == "latentmas-qwen3.5-4b" and T.PAPER_TAG not in other.tags


def test_smoke_runs_go_to_the_dev_project_and_folder():
    args = {**RUN_ARGS, "limit": 32}
    lay = T.layout_for("latentmas", args, T.RESULTS / "dev" / "smoke-gsm8k.jsonl", git="f438c94")
    assert (lay.project, lay.job_type, lay.folder) == ("latent-comm-dev", "smoke", "dev")
    assert lay.name.endswith(".s0.smoke") and "smoke" in lay.tags and T.PAPER_TAG not in lay.tags
    T.require_results_folder(lay)
    misfiled = T.layout_for("latentmas", args, LATENTMAS / "x.jsonl")
    with pytest.raises(T.TrackingError, match="latent/results/dev/"):
        T.require_results_folder(misfiled)
    with pytest.raises(T.TrackingError, match="latent/results/latentmas-qwen3-4b/"):
        T.require_results_folder(T.layout_for("latentmas", RUN_ARGS, T.RESULTS / "x.jsonl"))
    by_folder = T.layout_for("split", SPLIT_ARGS, T.RESULTS / "dev" / "try.jsonl")
    assert by_folder.project == "latent-comm-dev"


def test_layout_split_and_timing():
    probe = T.layout_for("split", SPLIT_ARGS, T.RESULTS / "split-probe" / "split-gsm8k.jsonl")
    assert probe.name == "split-probe.gsm8k.latent+wrong+nocache+text.m40.s0"
    assert probe.job_type == "eval" and T.PAPER_TAG in probe.tags
    timing = T.layout_for("split", {**SPLIT_ARGS, "n": 20, "batch_size": 1},
                          T.RESULTS / "split-probe" / "split-gsm8k-timing.jsonl")
    assert (timing.project, timing.job_type) == ("latent-comm", "timing")
    assert timing.name == probe.name + ".timing" and timing.run_id != probe.run_id
    nocache = T.layout_for("split", {**SPLIT_ARGS, "conditions": "nocache"},
                           T.RESULTS / "split-probe" / "n.jsonl")
    assert nocache.name == "split-probe.gsm8k.nocache.s0"


def test_layout_masbench():
    args = {"arms": "latent,wrong,nocache,text,both", "limit": None, "level": None, "m": 40,
            "seed": 0, "model": "Qwen/Qwen3.8-27B", "batch_size": 1, "max_new_tokens": 4096,
            "temperature": 0.0, "top_p": 1.0, "think": True, "report": None,
            "out": "latent/results/masbench/reasoning-qwen3.8-27b.jsonl"}
    assert T.infer_kind(args) == "masbench"
    lay = T.layout_for("masbench", args, T.RESULTS / "masbench" / "reasoning-qwen3.8-27b.jsonl")
    assert lay.name == "masbench.reasoning.latent+wrong+nocache+text+both.m40.s0"
    assert (lay.project, lay.group, lay.job_type) == ("latent-comm", "masbench", "eval")
    assert "qwen3.8-27b" in lay.tags and "reasoning" in lay.tags and T.PAPER_TAG not in lay.tags
    smoke = T.layout_for("masbench", {**args, "limit": 2, "level": "hard"},
                         T.RESULTS / "dev" / "masbench-smoke.jsonl")
    assert smoke.project == "latent-comm-dev" and smoke.job_type == "smoke"
    assert smoke.name == "masbench.reasoning.latent+wrong+nocache+text+both.m40.hard.s0.smoke"


def test_masbench_summary(tmp_path):
    out = tmp_path / "masbench-x.jsonl"
    meta = {"meta": {"arms": "latent,nocache", "level": None, "limit": None, "m": 8, "seed": 0,
                     "model": "Qwen/Qwen3.5-2B"}, "git": "abc"}
    rows = [{"qid": "q1", "level": "easy", "condition": "latent", "f1": 1.0, "answer": "x",
             "tokens": 10, "time_solver_s": 1.0, "message_bytes": 100},
            {"qid": "q2", "level": "hard", "condition": "latent", "f1": 0.0, "answer": None,
             "tokens": 20, "time_solver_s": 2.0, "message_bytes": 200},
            {"qid": "q1", "level": "easy", "condition": "nocache", "f1": 0.5, "answer": "y",
             "tokens": 30, "time_solver_s": 3.0}]
    out.write_text("\n".join(json.dumps(r) for r in [meta, *rows]) + "\n")
    s = T.results_summary("masbench", [out])
    assert s["latent/f1"] == 50.0 and s["latent/f1_easy"] == 100.0 and s["latent/f1_hard"] == 0.0
    assert s["latent/answered"] == 1 and s["latent/message_bytes"] == 200
    assert s["nocache/f1"] == 50.0 and "nocache/message_bytes" not in s


def test_layout_cacheback_stages():
    folder = T.RESULTS / "cacheback-fanoutqa"
    arm = T.layout_for("cacheback", CB_ARGS, folder / "cb-cb4.jsonl", git="b0ebfb4")
    assert arm.name == "cacheback-fanoutqa.fanoutqa.cb4.s0"
    assert arm.tags == ("git-b0ebfb4", "qwen3-8b", "fanoutqa")
    assert arm.artifact == "results-cacheback-fanoutqa-fanoutqa"
    text = T.layout_for("cacheback", {**CB_ARGS, "stage": "text", "shard": "1/4"},
                        folder / "cb-text-reports.shard1of4.jsonl")
    assert text.name == "cacheback-fanoutqa.fanoutqa.text-reports.s0.shard1of4"
    produce = T.layout_for("cacheback", {**CB_ARGS, "stage": "produce"}, folder / "cb-bank")
    assert produce.name == "cacheback-fanoutqa.fanoutqa.produce" and produce.artifact is None
    T.require_results_folder(produce)


def test_run_ids_are_deterministic_and_distinct():
    a = T.layout_for("latentmas", RUN_ARGS, LATENTMAS / "a.jsonl")
    assert a.run_id == T.layout_for("latentmas", RUN_ARGS, LATENTMAS / "a.jsonl").run_id
    assert a.run_id != T.layout_for("latentmas", RUN_ARGS, LATENTMAS / "b.jsonl").run_id
    assert T.run_id(T.PROJECT, a.out) != T.run_id(T.DEV_PROJECT, a.out)
    assert len(a.run_id) == 16 and a.run_id.isalnum()


def test_chain_conditions_match_the_runner():
    assert set(T.CHAIN) == set(run_mod.CHAIN_CONDITIONS)
    assert T.infer_kind(RUN_ARGS) == "latentmas" and T.infer_kind(SPLIT_ARGS) == "split"
    assert T.infer_kind(CB_ARGS) == "cacheback"


# ---------------------------------------------------------------- summaries and tables


def _write(path, meta, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in [meta, *rows]) + "\n", encoding="utf-8")
    return path


def _latentmas_file(path, args=RUN_ARGS, n=2):
    rows = []
    for cond, m in (("single", 0), ("latent", 40)):
        for i in range(n):
            rows.append({"qid": f"q{i}", "bench": "gsm8k", "condition": cond, "seed": 0, "m": m,
                         "answer": "1", "gold": "1", "correct": i == 0, "tokens": 10,
                         "time_s": 1.0, "batch": 0, "batch_size": n, "peak_mem_gb": 2.0,
                         "partner": None})
    rows.append({**rows[1], "correct": True})  # a resumed batch repeats a row: the last counts
    return _write(path, {"meta": args, "git": "abc1234", "host": "h", "gpu": "g",
                         "torch": "t"}, rows)


def _split_file(path, args=SPLIT_ARGS, questions=20):
    rows = []
    for i in range(questions):
        for c, ok in (("latent", 1), ("wrong", i % 2), ("nocache", 0), ("text", 1)):
            row = {"qid": f"q{i}", "condition": c, "seed": 0, "m": 40, "correct": ok,
                   "tokens": 10, "flops_sender": 1e12, "flops_solver_prompt": 1e12,
                   "flops_solver_answer": 1e12, "time_sender_s": 0.5, "time_solver_s": 2.0,
                   "time_solver_prompt_s": None if c == "text" else 0.1, "partner": None,
                   "answer": "x", "gold": "y"}
            if c != "nocache":
                row.update(message_bytes=2**20, message_tokens=50)
            rows.append(row)
    return _write(path, {"meta": args, "git": "76a2743", "questions": questions,
                         "skipped_unsplittable": 0}, rows)


def test_latentmas_summary_dedups_and_names_by_condition(tmp_path):
    p = _latentmas_file(tmp_path / "f.jsonl")
    s = T.results_summary("latentmas", [p])
    assert s["rows"] == 4 and s["single/acc"] == 100.0 and s["latent.m40/acc"] == 50.0
    assert s["single/questions"] == 2 and "single/acc_sd" not in s


def test_split_summary_has_paired_differences(tmp_path):
    s = T.results_summary("split", [_split_file(tmp_path / "s.jsonl")])
    assert s["latent/acc"] == 100.0 and s["wrong/acc"] == 50.0 and s["nocache/acc"] == 0.0
    assert s["diff/latent-wrong"] == 50.0 and s["diff/latent-nocache"] == 100.0
    assert s["kill_criterion_passed"] is True and "nocache/message_bytes" not in s


def test_cacheback_summary_per_arm_and_reports(tmp_path):
    rows = [{"qid": f"q{i}", "arm": "cb4", "kind": "cb", "ratio": 4, "draw": d,
             "strict": float(i == 0), "loose": 0.5, "tokens": 100, "closer": False,
             "unclosed": False, "time_s": 1.0, "batch_size": 3, "peak_mem_gb": 1.0,
             "msg_positions": 10, "msg_bytes": 2**20, "prompt_tokens": 5, "partner": None,
             "recall": [1, 0], "sender_positions": [10], "report_tokens": None,
             "sender_s": 1.0, "keeps": [[0, 1]] if d == 0 else None}
            for i in range(2) for d in range(3)]
    p = _write(tmp_path / "cb-cb4.jsonl", {"meta": CB_ARGS}, rows)
    s = T.results_summary("cacheback", [p])
    assert s["cb4/strict"] == 50.0 and s["cb4/recall"] == 50.0 and "cb4/sd" in s
    cols, data = T.table_rows("cacheback", [p])
    assert "keeps" not in cols and data[0][cols.index("recall")] == "[1, 0]"
    reports = [{"qid": "q0", "worker": w, "report": "long", "raw": "longer", "tokens": 7,
                "closer": w == 0, "time_s": 2.0, "batch_size": 3, "peak_mem_gb": 1.0}
               for w in range(3)]
    r = _write(tmp_path / "cb-text-reports.jsonl", {"meta": {**CB_ARGS, "stage": "text"}},
               reports)
    s = T.results_summary("cacheback", [r])
    assert s["reports"] == 3 and s["report_tokens"] == 7
    assert "report" not in T.table_rows("cacheback", [r])[0]


def test_table_rows_leave_out_texts(tmp_path):
    p = _split_file(tmp_path / "s.jsonl")
    lines = p.read_text().splitlines()
    lines[1] = json.dumps({**json.loads(lines[1]), "text": "long", "message": "long"})
    p.write_text("\n".join(lines))
    cols, data = T.table_rows("split", [p])
    assert "text" not in cols and "message" not in cols and len(data) == 80
    assert all(len(row) == len(cols) for row in data)


# ---------------------------------------------------------------- session (fake wandb)


class FakeRun:
    def __init__(self, **kw):
        self.kw, self.id, self.summary = kw, kw["id"], {}
        self.url = f"https://wandb.ai/me/{kw['project']}/runs/{kw['id']}"
        self.logged, self.artifacts, self.exit_code = [], [], None

    def log(self, d):
        self.logged.append(d)

    def log_artifact(self, art, aliases):
        self.artifacts.append((art, aliases))

    def finish(self, exit_code=0):
        self.exit_code = exit_code


@pytest.fixture
def fake_wandb(monkeypatch):
    mod = types.ModuleType("wandb")
    mod.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    mod.runs, mod.logins = [], []
    mod.Settings = lambda **kw: ("Settings", kw)

    def init(**kw):
        mod.runs.append(FakeRun(**kw))
        return mod.runs[-1]

    mod.init = init
    mod.login = lambda **kw: mod.logins.append(kw) or True

    class Table:
        def __init__(self, columns, data, allow_mixed_types):
            self.columns, self.data = columns, data

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata, self.files = name, type, metadata, []

        def add_file(self, path, name):
            self.files.append(name)

    mod.Table, mod.Artifact = Table, Artifact
    monkeypatch.setitem(sys.modules, "wandb", mod)
    for k, v in GOOD_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    return mod


def test_tracked_run_is_online_resumable_and_logs_results(fake_wandb, tmp_path):
    p = _split_file(tmp_path / "split-probe" / "s.jsonl")
    lay = T.layout_for("split", SPLIT_ARGS, p, git="76a2743")
    with T.TrackedRun(lay, {"seed": 0}) as run:
        run.log({"latent/running_acc": 50.0})
        run.log_results([p])
    kw = fake_wandb.runs[0].kw
    assert kw["mode"] == "online" and kw["resume"] == "allow" and kw["id"] == lay.run_id
    assert kw["settings"] == ("Settings", {"base_url": T.BASE_URL})
    assert (kw["project"], kw["group"], kw["name"]) == (lay.project, lay.group, lay.name)
    assert fake_wandb.logins == [{"host": T.BASE_URL, "verify": True}]
    r = fake_wandb.runs[0]
    assert r.exit_code == 0 and r.summary["latent/acc"] == 100.0
    assert len(r.logged[-1]["results_table"].data) == 80
    art, aliases = r.artifacts[0]
    assert (art.name, art.type, art.files, aliases) == (lay.artifact, "results", ["s.jsonl"],
                                                        [lay.name])


def test_tracked_run_fails_on_exception_and_refuses_offline(fake_wandb, monkeypatch):
    lay = T.layout_for("latentmas", RUN_ARGS, LATENTMAS / "x.jsonl")
    with pytest.raises(ValueError, match="boom"):
        with T.TrackedRun(lay, {}):
            raise ValueError("boom")
    assert fake_wandb.runs[0].exit_code == 1
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(T.TrackingError):
        T.TrackedRun(lay, {})
    assert len(fake_wandb.runs) == 1


# ---------------------------------------------------------------- registry and backfill


def test_registry_entries_from_files(tmp_path):
    split_file = _split_file(tmp_path / "split-probe" / "split-gsm8k.jsonl")
    smoke = _latentmas_file(tmp_path / "dev" / "smoke-gsm8k.jsonl",
                            {**RUN_ARGS, "limit": 2, "conditions": "single,latent",
                             "think": False, "out": "latent/results/smoke-gsm8k.jsonl"})
    notebook = ("Data: `latent/results/split-probe/split-gsm8k.jsonl` (job 21812868, H100).\n"
                "(`split-gsm8k-timing.jsonl`, job 21812869)\n")
    dates = wb.parse_git_log("date:2026-10-13\nlatent/results/split-gsm8k.jsonl\n\n"
                             "date:2026-10-10\nlatent/results/smoke-gsm8k.jsonl\n")
    entries = wb.build_registry([split_file, smoke], [], dates, notebook)
    s, d = entries
    assert s["run_key"] == "split-probe/split-gsm8k" and s["slurm_job"] == 21812868
    assert s["status"] == "complete" and s["rows"] == s["expected_rows"] == 80
    assert s["wandb_group"] == "split-probe" and s["wandb_url"] is None
    assert s["date"] == "2026-10-13"  # found under the path the meta row was written with
    assert d["wandb_project"] == "latent-comm-dev" and d["wandb_job_type"] == "smoke"
    assert d["status"] == "complete" and d["rows"] == 4 and d["expected_rows"] == 4
    assert d["date"] == "2026-10-10" and "since moved" in d["notes"]
    assert "--no-think" in d["command"] and "--limit 2" in d["command"]
    assert "--no-realign" not in d["command"] and d["slurm_job"] is None
    kept = wb.build_registry([split_file], [{**s, "wandb_url": "https://x", "notes": "hand"}],
                             {}, "")
    assert kept[0]["wandb_url"] == "https://x" and kept[0]["notes"] == "hand"


def test_committed_registry_matches_the_results_files():
    committed = {e["run_key"]: e for e in wb.read_registry()}
    rebuilt = wb.build_registry(wb.results_files(), [], {}, "")
    assert rebuilt, "no results files found"
    missing = sorted({e["run_key"] for e in rebuilt} - set(committed))
    assert not missing, f"registry out of date, run wandb_backfill --build-registry: {missing}"
    names = [(e["wandb_project"], e["wandb_name"]) for e in rebuilt]
    assert len(set(names)) == len(names)
    for e in rebuilt:
        c = committed[e["run_key"]]
        for k in ("files", "status", "rows", "command", "wandb_project", "wandb_group",
                  "wandb_job_type", "wandb_name"):
            assert c[k] == e[k], (e["run_key"], k)
        folder = e["run_key"].split("/")[0]
        assert folder == ("dev" if e["wandb_project"] == T.DEV_PROJECT else e["wandb_group"])


def test_backfill_dry_run_opens_no_run(fake_wandb, tmp_path, capsys, monkeypatch):
    fake_wandb.init = fake_wandb.login = None  # any call would fail
    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setenv("WANDB_MODE", "offline")  # the dry run needs no W&B setup
    monkeypatch.setattr(T, "REPO", tmp_path)
    p = _split_file(tmp_path / "latent" / "results" / "split-probe" / "split-gsm8k.jsonl")
    entries = wb.build_registry([p], [], {}, "")
    registry = tmp_path / "runs.jsonl"
    wb.write_registry(entries, registry)
    done = wb.backfill(wb.read_registry(registry), dry_run=True, only=None, registry=registry)
    out = capsys.readouterr().out
    assert len(done) == 1 and "table results_table: 80 rows" in out
    assert "latent-comm/\n  split-probe/\n    eval/\n      split-probe.gsm8k." in out
    assert "results-split-probe-gsm8k  (1 version)" in out
    assert wb.read_registry(registry)[0]["wandb_url"] is None
