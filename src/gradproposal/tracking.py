"""Experiment tracking: hosted W&B cloud only, one fixed layout, results logged from the files.

Sections (wandb itself is imported only inside the session functions, so the guard, the layout
and the summaries are plain functions that unit tests call without W&B or a network):

  guard      `require_wandb()` aborts unless W&B can log to the hosted cloud: WANDB_MODE online,
             base URL https://api.wandb.ai, credentials present, wandb installed. Every GPU
             entrypoint calls it right after parsing arguments, before data or model loading.
  layout     `layout_for()` maps a results file to its W&B project, group, job type, run name,
             tags, artifact collection and run id, and to its local folder
             latent/results/<group>/ (the folder name is the W&B group; smoke runs go to dev/).
  summaries  `results_summary()` and `table_rows()` read a results file; live runs and the
             backfill log the same numbers.
  session    `TrackedRun`: online-only `wandb.init` with a deterministic id and resume="allow",
             so a requeued Slurm job continues its run. Exceptions finish the run as failed.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import statistics
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "latent" / "results"
WANDB_DIR = REPO / "runs"  # gitignored; W&B's local run files

# ---------------------------------------------------------------- guard

BASE_URL = "https://api.wandb.ai"
_FALSE = ("", "0", "false", "no", "off")


class TrackingError(RuntimeError):
    """W&B cloud tracking is unavailable or a run is filed in the wrong place."""


def netrc_path(env: Mapping[str, str]) -> Path:
    """The netrc file wandb reads: $NETRC if set, else ~/.netrc (~/_netrc on Windows)."""
    if env.get("NETRC"):
        return Path(env["NETRC"]).expanduser()
    home = Path.home()
    return home / ("_netrc" if os.name == "nt" and not (home / ".netrc").exists() else ".netrc")


def netrc_has_wandb(path: Path) -> bool:
    """True if the file has `machine api.wandb.ai` with a password. Scans tokens only and never
    raises or echoes file content (parse errors from the netrc module can quote it)."""
    try:
        tokens = Path(path).read_text(encoding="utf-8", errors="replace").split()
    except OSError:
        return False
    for i, tok in enumerate(tokens[:-1]):
        if tok == "machine" and tokens[i + 1].split(":")[0] == "api.wandb.ai":
            for j in range(i + 2, len(tokens) - 1):
                if tokens[j] in ("machine", "default"):
                    break
                if tokens[j] == "password" and tokens[j + 1]:
                    return True
    return False


def wandb_env_errors(env: Mapping[str, str], netrc_ok: bool,
                     wandb_installed: bool = True) -> list[str]:
    """Reasons W&B cannot log to the hosted cloud (empty list: it can). Pure: no I/O.

    WANDB_MODE must be unset (then forced to online) or exactly "online"; offline, disabled,
    dryrun, shared, empty and unknown modes are refused. WANDB_BASE_URL must be unset (then
    pinned) or https://api.wandb.ai. Credentials: WANDB_API_KEY or a netrc entry."""
    errors = []
    if not wandb_installed:
        errors.append("the wandb package is not installed (run `uv sync`)")
    mode = env.get("WANDB_MODE")
    if mode is not None and mode != "online":
        errors.append(f"WANDB_MODE={mode!r}: only 'online' (hosted W&B cloud) is allowed")
    if env.get("WANDB_DISABLED", "").strip().lower() not in _FALSE:
        errors.append("WANDB_DISABLED is set: W&B logging cannot be disabled")
    url = env.get("WANDB_BASE_URL")
    if url is not None and url.rstrip("/") != BASE_URL:
        errors.append(f"WANDB_BASE_URL={url!r}: only {BASE_URL} is allowed")
    if not env.get("WANDB_API_KEY", "").strip() and not netrc_ok:
        errors.append("no W&B credentials: set WANDB_API_KEY, or point NETRC at a netrc file "
                      "with `machine api.wandb.ai` (`wandb login` writes ~/.netrc)")
    return errors


def require_wandb(env: MutableMapping[str, str] | None = None) -> None:
    """Abort (TrackingError) unless W&B can log to the hosted cloud; then pin WANDB_MODE=online
    and WANDB_BASE_URL in `env` (default os.environ). Call before loading data or models."""
    env = os.environ if env is None else env
    errors = wandb_env_errors(env, netrc_has_wandb(netrc_path(env)),
                              importlib.util.find_spec("wandb") is not None)
    if errors:
        raise TrackingError("W&B cloud tracking is required for this run; refusing to start:\n"
                            + "\n".join(f"  - {e}" for e in errors))
    env["WANDB_MODE"] = "online"
    env["WANDB_BASE_URL"] = BASE_URL


# ---------------------------------------------------------------- layout
#
# project   latent-comm: real experiment runs; latent-comm-dev: smoke runs (--limit on the
#           LatentMAS runner, an out file in latent/results/dev/ or a stem starting "smoke").
# group     experiment family = local folder latent/results/<group>/:
#           latentmas-<model> (run.py), split-probe (split.py), cacheback-fanoutqa (cb_run.py),
#           masbench (masbench.py).
# job_type  eval; timing (split probe at batch size 1); smoke.
# name      <group>.<bench>.<conditions or arm>[.m<m>].s<seed>[.<job_type> unless eval];
#           lists joined by "+", e.g. latentmas-qwen3-4b.gsm8k.latent+wrong+noise+nocache.m40.s0
# tags      git-<sha7> (each commit the file was produced with), model, bench, paper-p1 (real
#           runs feeding Paper 1), smoke, backfill (logged after the fact from the file).
# artifact  results-<group>-<bench>, type results; each run adds a version aliased by its name.
# id        sha256("<project>:<repo-relative results path>")[:16]: a requeued job (same --out)
#           resumes its run.

PROJECT = "latent-comm"
DEV_PROJECT = "latent-comm-dev"
DEV_FOLDER = "dev"
PAPER_TAG = "paper-p1"
PAPER_GROUPS = ("latentmas-qwen3-4b", "split-probe")
KINDS = {  # results kind -> python module that writes it
    "latentmas": "gradproposal.latent.run",
    "split": "gradproposal.latent.split",
    "cacheback": "gradproposal.latent.cb_run",
    "masbench": "gradproposal.latent.masbench",
}
CHAIN = ("latent", "wrong", "noise")  # conditions that take latent steps (run.CHAIN_CONDITIONS)
SPLIT_CHAIN = ("latent", "wrong")  # split arms that take latent steps


@dataclass(frozen=True)
class RunLayout:
    kind: str
    project: str
    group: str
    job_type: str
    name: str
    tags: tuple[str, ...]
    artifact: str | None  # results collection; None when the run writes no results file
    out: str  # repo-relative POSIX path of the results file (or bank folder)
    folder: str  # its expected folder under latent/results: the group, or dev

    @property
    def run_id(self) -> str:
        return run_id(self.project, self.out)


def rel_path(path) -> str:
    """Repo-relative POSIX path (absolute if outside the repo); the same on HPC and the desktop."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def run_id(project: str, out: str) -> str:
    return hashlib.sha256(f"{project}:{out}".encode()).hexdigest()[:16]


def model_short(model: str) -> str:
    return model.rsplit("/", 1)[-1].lower()


def infer_kind(args: Mapping) -> str:
    """Which entrypoint wrote a meta row, from its argument names."""
    if "stage" in args:
        return "cacheback"
    if "arms" in args and "level" in args:
        return "masbench"
    if "partner_group" in args:
        return "latentmas"
    if "report" in args:
        return "split"
    raise ValueError(f"unrecognized results metadata: {sorted(args)}")


def _plus(value) -> str:
    return "+".join(str(value).split(","))


def is_smoke(kind: str, args: Mapping, out: Path) -> bool:
    return (out.parent.name == DEV_FOLDER or out.stem.startswith("smoke")
            or (kind == "latentmas" and args.get("limit") is not None))


def layout_for(kind: str, args: Mapping, out, git: str | Iterable[str] | None = None,
               backfill: bool = False) -> RunLayout:
    """The W&B layout of one results file (or, for CacheBack `produce`, the bank folder), from
    the entrypoint's arguments (the meta row's "meta" dict). Deterministic."""
    out = Path(out)
    smoke = is_smoke(kind, args, out)
    if kind == "latentmas":
        group, bench = f"latentmas-{model_short(args['model'])}", args["bench"]
        conds = args["conditions"].split(",")
        parts = [bench, "+".join(conds)]
        if any(c in CHAIN for c in conds):
            parts.append(f"m{_plus(args['m'])}")
        parts.append(f"s{_plus(args['seeds'])}")
        job_type = "smoke" if smoke else "eval"
    elif kind == "split":
        group, bench = "split-probe", "gsm8k"
        conds = args["conditions"].split(",")
        parts = [bench, "+".join(conds)]
        if any(c in SPLIT_CHAIN for c in conds):
            parts.append(f"m{args['m']}")
        parts.append(f"s{args['seed']}")
        job_type = "smoke" if smoke else "timing" if args.get("batch_size") == 1 else "eval"
    elif kind == "masbench":
        group, bench = "masbench", "reasoning"
        arms = args["arms"].split(",")
        parts = [bench, "+".join(arms)]
        if any(c in SPLIT_CHAIN for c in arms):
            parts.append(f"m{args['m']}")
        if args.get("level"):
            parts.append(args["level"])
        parts.append(f"s{args['seed']}")
        job_type = "smoke" if smoke or args.get("limit") else "eval"
    elif kind == "cacheback":
        group, bench = "cacheback-fanoutqa", "fanoutqa"
        stage = args["stage"]
        if stage == "produce":
            parts = [bench, "produce"]
        elif stage == "text":
            k, n = map(int, args["shard"].split("/"))
            parts = [bench, "text-reports", f"s{args['seed']}"] + ([f"shard{k}of{n}"] if n > 1
                                                                    else [])
        else:
            parts = [bench, out.stem.removeprefix("cb-"), f"s{args['seed']}"]
        job_type = "smoke" if smoke else "eval"
    else:
        raise ValueError(kind)
    if job_type != "eval":
        parts.append(job_type)
    shas = [git] if isinstance(git, str) or git is None else list(git)
    tags = [f"git-{s[:7]}" for s in shas if s and s != "unknown"]
    tags += [model_short(args["model"]), bench]
    if group in PAPER_GROUPS and not smoke:
        tags.append(PAPER_TAG)
    if smoke:
        tags.append("smoke")
    if backfill:
        tags.append("backfill")
    produce = kind == "cacheback" and args["stage"] == "produce"
    return RunLayout(
        kind=kind, project=DEV_PROJECT if smoke else PROJECT, group=group, job_type=job_type,
        name=".".join([group, *parts]), tags=tuple(dict.fromkeys(tags)),
        artifact=None if produce else f"results-{group}-{bench}", out=rel_path(out),
        folder=DEV_FOLDER if smoke else group)


def require_results_folder(layout: RunLayout) -> None:
    """Live runs write into latent/results/<group>/ (smoke runs: latent/results/dev/), the local
    mirror of the W&B layout. Raises before any data or model loading."""
    want = (RESULTS / layout.folder).resolve()
    got = (REPO / layout.out).resolve().parent
    if got != want:
        raise TrackingError(
            f"{layout.out} is filed outside its folder: write it to "
            f"{rel_path(want)}/ (W&B project {layout.project}, group {layout.group}, "
            f"job type {layout.job_type})")


def config_from_meta(meta_row: Mapping) -> dict:
    """W&B config: the CLI arguments plus run metadata (git, host, GPU, versions), flattened."""
    extra = {k: v for k, v in meta_row.items() if k not in ("meta", "wandb")}
    return {**meta_row["meta"], **extra}


# ---------------------------------------------------------------- summaries

LONG_FIELDS = ("text", "message", "raw", "report", "keeps")  # left out of W&B tables


def read_results(path) -> tuple[dict | None, list[dict]]:
    """(first meta row or None, data rows) of a results JSONL file."""
    meta, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "meta" in r:
                meta = meta or r
            else:
                rows.append(r)
    return meta, rows


def _key(kind: str, r: dict) -> tuple:
    if kind == "latentmas":
        return r["bench"], r["qid"], r["condition"], r["seed"], r["m"]
    if kind in ("split", "masbench"):
        return r["qid"], r["condition"]
    if "worker" in r:
        return r["qid"], r["worker"]
    return r["qid"], r["arm"], r["draw"]


def unique_rows(kind: str, paths) -> list[dict]:
    """Data rows of the files; a resumed batch can repeat rows, the last one per key is kept."""
    keyed = {}
    for p in paths:
        for r in read_results(p)[1]:
            keyed[_key(kind, r)] = r
    return list(keyed.values())


def _median(xs):
    """Upper median, as split.report computes it."""
    xs = sorted(xs)
    return xs[len(xs) // 2]


def results_summary(kind: str, paths) -> dict:
    """W&B summary values for results files (accuracy per condition and m, paired contrasts)."""
    paths = [Path(p) for p in paths]
    rows = unique_rows(kind, paths)
    out: dict = {"rows": len(rows)}
    if not rows:
        return out
    if kind == "latentmas":
        from gradproposal.latent.report import summarize

        summary = summarize(rows)
        several = len({b for b, _, _ in summary}) > 1
        for (bench, cond, m), v in sorted(summary.items()):
            key = (f"{bench}/" if several else "") + (f"{cond}.m{m}" if cond in CHAIN else cond)
            out[f"{key}/acc"] = v["acc"]
            if v["seeds"] > 1:
                out[f"{key}/acc_sd"] = v["sd"]
            out.update({f"{key}/questions": v["n"], f"{key}/seeds": v["seeds"],
                        f"{key}/tokens": v["tokens"], f"{key}/time_s": v["time"],
                        f"{key}/peak_mem_gb": v["mem"]})
    elif kind == "split":
        from gradproposal.latent.split import report

        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["condition"], []).append(r)
        for cond, rs in by.items():
            out[f"{cond}/acc"] = 100 * sum(r["correct"] for r in rs) / len(rs)
            out[f"{cond}/questions"] = len(rs)
            out[f"{cond}/answer_tokens"] = _median([r["tokens"] for r in rs])
            out[f"{cond}/sender_tflops"] = _median([r["flops_sender"] for r in rs]) / 1e12
            out[f"{cond}/total_tflops"] = _median(
                [r["flops_sender"] + r["flops_solver_prompt"] + r["flops_solver_answer"]
                 for r in rs]) / 1e12
            out[f"{cond}/time_sender_s"] = _median([r["time_sender_s"] for r in rs])
            out[f"{cond}/time_solver_s"] = _median([r["time_solver_s"] for r in rs])
            if "message_bytes" in rs[0]:
                out[f"{cond}/message_bytes"] = _median([r["message_bytes"] for r in rs])
                out[f"{cond}/message_tokens"] = _median([r["message_tokens"] for r in rs])
        if len(paths) == 1:
            with contextlib.redirect_stdout(io.StringIO()):  # report() prints its table
                diffs = report(str(paths[0]))
            for name, (mean, lo, hi) in diffs.items():
                out.update({f"diff/{name}": mean, f"diff/{name}_lo": lo, f"diff/{name}_hi": hi})
            if "latent-wrong" in diffs:
                out["kill_criterion_passed"] = diffs["latent-wrong"][0] >= 10
    elif kind == "masbench":
        from gradproposal.latent.masbench import summarize as mb_summarize

        for cond, d in mb_summarize(rows).items():
            out[f"{cond}/f1"] = d["f1"]
            out[f"{cond}/questions"] = d["n"]
            out[f"{cond}/answered"] = d["answered"]
            for level in ("easy", "hard"):
                if f"f1_{level}" in d:
                    out[f"{cond}/f1_{level}"] = d[f"f1_{level}"]
            rs = [r for r in rows if r["condition"] == cond]
            out[f"{cond}/answer_tokens"] = _median([r["tokens"] for r in rs])
            out[f"{cond}/time_solver_s"] = _median([r["time_solver_s"] for r in rs])
            if "message_bytes" in rs[0]:
                out[f"{cond}/message_bytes"] = _median([r["message_bytes"] for r in rs])
    elif kind == "cacheback":
        if "worker" in rows[0]:  # text-sender reports
            out.update(reports=len(rows), report_tokens=_median([r["tokens"] for r in rows]),
                       closer_pct=100 * statistics.mean(r["closer"] for r in rows),
                       time_s=statistics.mean(r["time_s"] for r in rows))
        else:
            from gradproposal.latent.cb_report import summarize

            by = {}
            for r in rows:
                by.setdefault(r["arm"], []).append(r)
            for arm, rs in by.items():
                for k, v in summarize(rs).items():
                    if v is not None and v == v:  # drop None and NaN (one draw: no sd)
                        out[f"{arm}/{k}"] = v
    else:
        raise ValueError(kind)
    return out


def table_rows(kind: str, paths) -> tuple[list[str], list[list]]:
    """Per-question table (columns, data) without long texts; lists and dicts as JSON strings."""
    rows = unique_rows(kind, paths)
    columns = list(dict.fromkeys(k for r in rows for k in r if k not in LONG_FIELDS))
    data = [[json.dumps(v) if isinstance(v, (list, dict)) else v
             for v in (r.get(c) for c in columns)] for r in rows]
    return columns, data


# ---------------------------------------------------------------- session


def make_table(columns: list[str], data: list[list]):
    import wandb

    return wandb.Table(columns=columns, data=data, allow_mixed_types=True)


def make_artifact(layout: RunLayout, paths):
    """A new version of the run's results collection holding its results file(s)."""
    import wandb

    paths = [Path(p) for p in paths]
    art = wandb.Artifact(layout.artifact, type="results",
                         metadata={"run": layout.name, "files": [rel_path(p) for p in paths]})
    for p in paths:
        art.add_file(str(p), name=p.name)
    return art


class TrackedRun:
    """One online W&B run. No offline or disabled mode; any W&B error propagates and fails the
    run. Use as a context manager: an exception inside finishes the run with exit code 1."""

    def __init__(self, layout: RunLayout, config: Mapping):
        require_wandb()
        import wandb

        if not wandb.login(host=BASE_URL, verify=True):
            raise TrackingError(f"W&B login to {BASE_URL} failed")
        WANDB_DIR.mkdir(parents=True, exist_ok=True)
        self.layout = layout
        self.run = wandb.init(
            project=layout.project, group=layout.group, job_type=layout.job_type,
            name=layout.name, id=layout.run_id, resume="allow", tags=list(layout.tags),
            config=dict(config), mode="online", dir=str(WANDB_DIR),
            settings=wandb.Settings(base_url=BASE_URL))
        if self.run is None:
            raise TrackingError("wandb.init returned no run")
        print(f"W&B run {layout.project}/{layout.group}/{layout.name}: {self.run.url}",
              flush=True)

    def pointer(self) -> dict:
        """Written into the results file's meta row, so the registry can link the run."""
        return {"project": self.layout.project, "id": self.run.id, "url": self.run.url}

    def log(self, metrics: Mapping) -> None:
        self.run.log(dict(metrics))

    def summary(self, values: Mapping) -> None:
        self.run.summary.update(dict(values))

    def log_results(self, paths) -> None:
        """Summary values, the per-question table and the results artifact, from the files."""
        paths = [Path(p) for p in paths]
        self.summary(results_summary(self.layout.kind, paths))
        self.log({"results_table": make_table(*table_rows(self.layout.kind, paths))})
        if self.layout.artifact:
            self.run.log_artifact(make_artifact(self.layout, paths), aliases=[self.layout.name])

    def finish(self, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)

    def __enter__(self) -> TrackedRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=0 if exc_type is None else 1)
