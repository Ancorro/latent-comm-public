"""Run registry (latent/results/runs.jsonl) and W&B backfill of results logged before tracking.

    python -m gradproposal.latent.wandb_backfill --build-registry   # (re)build the registry
    python -m gradproposal.latent.wandb_backfill --dry-run          # print what would be logged
    python -m gradproposal.latent.wandb_backfill [--only RUN_KEY]   # log to W&B, record the URLs

Registry: one JSON object per results file latent/results/<group>/<name>.jsonl (one logical run),
built from the file's meta row (first line), its rows, latent/NOTEBOOK.md (Slurm job ids) and git
(the date of the commit that added the file, when the meta row has no start date). Fields that
cannot be recovered are null, never guessed. A rebuild keeps non-null `slurm_job`, `wandb_url`
and `notes` of existing entries, and takes `wandb_url` from the meta row of runs logged live.

Backfill: one W&B run per registry entry without a `wandb_url`, in the layout of
gradproposal.tracking (tag `backfill`): config from the meta row; summary, per-question table and
results artifact from the file, exactly as a live run logs them at its end. Each URL is written
back to the registry as soon as its run finishes. --dry-run builds the same tables and artifacts
locally and prints them with the project/group/job-type/run tree; it opens no W&B run and needs no
credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

from gradproposal import tracking as T

REGISTRY = T.RESULTS / "runs.jsonl"
NOTEBOOK = T.REPO / "latent" / "NOTEBOOK.md"
KEEP = ("slurm_job", "wandb_url", "notes")  # set by hand or by the backfill; kept on rebuild
GIT_LOG = ["git", "log", "--diff-filter=A", "--format=date:%cs", "--name-only", "--",
           "latent/results"]
NEGATED = {"think": "--no-think", "realign": "--no-realign"}  # store_false flags

# ---------------------------------------------------------------- registry


def results_files(root: Path = T.RESULTS) -> list[Path]:
    """Results files: latent/results/<group>/*.jsonl without the solver texts."""
    return sorted(p for p in root.glob("*/*.jsonl") if not p.name.endswith(".texts.jsonl"))


def parse_git_log(text: str) -> dict[str, str]:
    """{path: date of the earliest commit adding it} from the output of GIT_LOG."""
    dates, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("date:"):
            current = line.removeprefix("date:")
        elif line and current:
            dates[line] = current  # newest first: the last write is the earliest commit
    return dates


def git_add_dates() -> dict[str, str]:
    """Add dates from the local git history ({} without git or history)."""
    try:
        out = subprocess.run(GIT_LOG, cwd=T.REPO, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    return parse_git_log(out.stdout)


def notebook_jobs(text: str, names) -> dict[str, int]:
    """{file name: Slurm job id} for notebook lines naming exactly one job and the file."""
    out = {}
    for line in text.splitlines():
        jobs = re.findall(r"\bjob (\d{6,})\b", line)
        if len(jobs) != 1:
            continue
        for name in names:
            if name not in out and re.search(rf"(?<![\w.-]){re.escape(name)}", line):
                out[name] = int(jobs[0])
    return out


def command(kind: str, args: dict) -> str:
    """The python command line, reconstructed from the meta row's arguments."""
    parts = ["python", "-m", T.KINDS[kind]] + ([args["stage"]] if kind == "cacheback" else [])
    for k, v in args.items():
        if k == "stage" or v is None:
            continue
        if k in NEGATED:
            parts += [] if v else [NEGATED[k]]
        else:
            parts += [f"--{k.replace('_', '-')}", str(v)]
    return shlex.join(parts)


def expected_rows(kind: str, meta: dict) -> int | None:
    """Rows a finished run writes (unique keys), or None when it cannot be known."""
    from gradproposal.latent import bench as B

    a = meta["meta"]
    if kind == "latentmas":
        conds = a["conditions"].split(",")
        per_q = sum(len(a["m"].split(",")) if c in T.CHAIN else 1 for c in conds)
        return len(B.load(a["bench"], a["n"], a["limit"])) * len(a["seeds"].split(",")) * per_q
    if kind == "split":
        return meta["questions"] * len(a["conditions"].split(","))
    if kind == "cacheback" and a["stage"] == "receive":
        return len(B.load_fanoutqa(a["limit"])) * len(a["draws"].split(","))
    return None


def registry_entry(path: Path, dates: dict[str, str], jobs: dict[str, int]) -> dict:
    meta, _ = T.read_results(path)
    if meta is None:
        raise ValueError(f"{path}: no meta row")
    args = meta["meta"]
    kind = T.infer_kind(args)
    layout = T.layout_for(kind, args, path, git=meta.get("git"))
    rel, written_as = T.rel_path(path), args.get("out")
    rows, expected = len(T.unique_rows(kind, [path])), expected_rows(kind, meta)
    status = ("unknown" if expected is None else "complete" if rows == expected
              else "partial" if rows < expected else "inconsistent")
    notes = []
    date = (meta.get("date") or "")[:10] or dates.get(rel) or dates.get(written_as or "")
    if date and not meta.get("date"):
        notes.append("date is the commit that added the file (the run ended on or before it)")
    if written_as and written_as != rel:
        notes.append(f"written as {written_as}, since moved")
    if layout.folder != path.parent.name:
        notes.append(f"filed in {path.parent.name}/ but its layout folder is {layout.folder}/")
    return {
        "run_key": f"{path.parent.name}/{path.stem}", "experiment": layout.group,
        "date": date or None, "files": [rel], "git": meta.get("git"),
        "slurm_job": jobs.get(path.name), "host": meta.get("host"), "gpu": meta.get("gpu"),
        "command": command(kind, args), "status": status, "rows": rows,
        "expected_rows": expected, "wandb_project": layout.project,
        "wandb_group": layout.group, "wandb_job_type": layout.job_type,
        "wandb_name": layout.name, "wandb_url": (meta.get("wandb") or {}).get("url"),
        "notes": "; ".join(notes) or None,
    }


def build_registry(files, old: list[dict], dates: dict[str, str], notebook: str) -> list[dict]:
    jobs = notebook_jobs(notebook, [p.name for p in files])
    before = {e["run_key"]: e for e in old}
    out = []
    for p in files:
        e = registry_entry(p, dates, jobs)
        prev = before.get(e["run_key"], {})
        e.update({k: prev[k] for k in KEEP if prev.get(k) is not None})
        out.append(e)
    return out


def read_registry(path: Path = REGISTRY) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_registry(entries: list[dict], path: Path = REGISTRY) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- backfill


def plan(entry: dict) -> tuple[T.RunLayout, dict, list[Path]]:
    """Layout, config and files of one registry entry; the layout must match the registry."""
    paths = [T.REPO / f for f in entry["files"]]
    meta, _ = T.read_results(paths[0])
    layout = T.layout_for(T.infer_kind(meta["meta"]), meta["meta"], paths[0],
                          git=meta.get("git"), backfill=True)
    have = (layout.project, layout.group, layout.job_type, layout.name)
    want = tuple(entry[k] for k in ("wandb_project", "wandb_group", "wandb_job_type",
                                    "wandb_name"))
    if have != want:
        raise SystemExit(f"{entry['run_key']}: registry layout {want} differs from {have}; "
                         "rebuild it with --build-registry")
    return layout, T.config_from_meta(meta), paths


def describe(entry: dict, layout: T.RunLayout, config: dict, paths: list[Path]) -> None:
    """Print what the backfill logs for one entry, building the W&B objects locally."""
    summary = T.results_summary(layout.kind, paths)
    columns, data = T.table_rows(layout.kind, paths)
    T.make_table(columns, data)  # surfaces W&B type errors without opening a run
    if layout.artifact:
        T.make_artifact(layout, paths)
    print(f"\n{entry['run_key']}  ->  {layout.project} / {layout.group} / {layout.job_type} / "
          f"{layout.name}  (id {layout.run_id})")
    print(f"  tags: {', '.join(layout.tags)}")
    print(f"  config keys ({len(config)}): {', '.join(sorted(config))}")
    print("  summary: " + ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    print(f"  table results_table: {len(data)} rows x {len(columns)} columns")
    if layout.artifact:
        print(f"  artifact {layout.artifact}:{layout.name} (type results): "
              + ", ".join(T.rel_path(p) for p in paths))


def tree(layouts: list[T.RunLayout]) -> str:
    """project -> group -> job type -> run names, then the artifact collections."""
    nested = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    collections = defaultdict(lambda: defaultdict(int))
    for lay in layouts:
        nested[lay.project][lay.group][lay.job_type].append(lay.name)
        if lay.artifact:
            collections[lay.project][lay.artifact] += 1
    lines = []
    for project in sorted(nested):
        lines.append(f"{project}/")
        for group in sorted(nested[project]):
            lines.append(f"  {group}/")
            for job_type in sorted(nested[project][group]):
                lines.append(f"    {job_type}/")
                lines += [f"      {name}" for name in sorted(nested[project][group][job_type])]
        lines.append("  artifacts (type results):")
        lines += [f"    {name}  ({n} version{'s' * (n > 1)})"
                  for name, n in sorted(collections[project].items())]
    return "\n".join(lines)


def backfill(entries: list[dict], dry_run: bool, only: set[str] | None,
             registry: Path = REGISTRY) -> list[T.RunLayout]:
    unknown = (only or set()) - {e["run_key"] for e in entries}
    if unknown:
        raise SystemExit(f"not in the registry: {sorted(unknown)}")
    if not dry_run:
        T.require_wandb()
    done = []
    for e in entries:
        if only and e["run_key"] not in only:
            continue
        if e.get("wandb_url"):
            print(f"skip {e['run_key']}: already logged at {e['wandb_url']}")
            continue
        layout, config, paths = plan(e)
        if dry_run:
            describe(e, layout, config, paths)
        else:
            with T.TrackedRun(layout, config) as run:
                run.log_results(paths)
                e["wandb_url"] = run.run.url
            write_registry(entries, registry)
        done.append(layout)
    print(f"\n{'would log' if dry_run else 'logged'} {len(done)} run(s):\n{tree(done)}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-registry", action="store_true",
                    help="(re)build latent/results/runs.jsonl from the results files and exit")
    ap.add_argument("--git-log", help="file with the output of `git log --diff-filter=A "
                    "--format=date:%%cs --name-only -- latent/results` (checkouts without .git)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be logged")
    ap.add_argument("--only", nargs="+", metavar="RUN_KEY", help="registry run_key(s)")
    ap.add_argument("--registry", default=str(REGISTRY))
    a = ap.parse_args(argv)
    registry = Path(a.registry)
    if a.build_registry:
        dates = (parse_git_log(Path(a.git_log).read_text(encoding="utf-8")) if a.git_log
                 else git_add_dates())
        notebook = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
        entries = build_registry(results_files(), read_registry(registry), dates, notebook)
        write_registry(entries, registry)
        print(f"wrote {len(entries)} entries to {T.rel_path(registry)}")
        return
    entries = read_registry(registry)
    if not entries:
        raise SystemExit(f"{registry} is empty: run with --build-registry first")
    backfill(entries, a.dry_run, set(a.only) if a.only else None, registry)


if __name__ == "__main__":
    main()
