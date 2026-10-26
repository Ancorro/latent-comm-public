"""FanOutQA scoring and message recall, ported from the CacheBack code (github.com/maxr0ssi/rclc,
tag paper-v1, Apache-2.0): `rcc/benchmarks/fanoutqa/scoring_core.py`, `scoring_equivalence.py`
and `rcc/run/handoff_recall.py`. The reference patch and alias list are its JSON files, copied to
`latent/data/fanoutqa/`.

A gold answer flattens to reference strings (dict keys and values, list items). Each becomes a
group of accepted forms; the patch fixes 9 references contradicted by the pinned pages, and the
equivalence rules accept "72.0" for "72", "km2" for "km²" and two aliases. Strict = every group
appears in the visible answer, loose = fraction of groups.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from functools import lru_cache

from gradproposal.latent.bench import DATA

_FILES = DATA / "fanoutqa"
Groups = tuple[tuple[str, ...], ...]


def gold_leaves(answer) -> tuple[str, ...]:
    def flat(v) -> list[str]:
        if isinstance(v, bool):
            return ["yes" if v else "no"]
        if isinstance(v, dict):
            return [str(k) for k in v] + [x for c in v.values() for x in flat(c)]
        if isinstance(v, (list, tuple)):
            return [x for c in v for x in flat(c)]
        return [str(v)]

    return tuple(flat(answer))


def _norm(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. ]", " ", text)).strip()


def _forms(leaf: str) -> tuple[str, ...]:
    forms = {_norm(leaf)}
    cleaned = leaf.replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        pass
    else:
        forms.add(_norm(cleaned))
        if number == int(number):
            forms.add(_norm(str(int(number))))
    return tuple(f for f in forms if f)


_INTEGER_DECIMAL = re.compile(r"(?<![\w.])(\d+)\.0+(?!\w|\.\w)")
_SQUARE_UNIT = re.compile(r"(?<![^\W\d])(mm|cm|km|m)²(?!\w)")


def normalize_notation(text: str) -> str:
    """Integer decimals and squared metric units, canonicalized without rounding."""
    return _SQUARE_UNIT.sub(r"\g<1>2", _INTEGER_DECIMAL.sub(r"\1", text.lower()))


@lru_cache(maxsize=1)
def _patch() -> dict[str, dict[int, tuple[str, Groups]]]:
    payload = json.loads((_FILES / "gold_patch.json").read_text(encoding="utf-8"))
    out: dict[str, dict[int, tuple[str, Groups]]] = {}
    for e in payload["entries"]:
        groups = tuple(tuple(str(x) for x in g) for g in e["groups"])
        out.setdefault(str(e["qid"]), {})[int(e["leaf_index"])] = (str(e["original"]), groups)
    return out


@lru_cache(maxsize=1)
def _aliases() -> dict[str, list[tuple[int, str, tuple[str, ...]]]]:
    payload = json.loads((_FILES / "equivalence_aliases.json").read_text(encoding="utf-8"))
    out: dict[str, list[tuple[int, str, tuple[str, ...]]]] = {}
    for e in payload["entries"]:
        out.setdefault(e["qid"], []).append(
            (int(e["leaf_index"]), e["original"], tuple(e["alternatives"])))
    return out


def gold_groups(qid: str, answer) -> Groups:
    """Patched reference groups with the audited aliases and notation rules applied."""
    leaves = gold_leaves(answer)
    groups: list[tuple[str, ...]] = []
    patch = _patch().get(qid, {})
    for i, leaf in enumerate(leaves):
        if i in patch:
            original, replacement = patch[i]
            assert original == leaf, (qid, i, original, leaf)
            groups.extend(replacement)
        else:
            groups.append((leaf,))
    for i, original, alternatives in _aliases().get(qid, []):
        assert leaves[i] == original and groups.count((original,)) == 1
        groups[groups.index((original,))] = (original, *alternatives)
    return tuple(tuple(normalize_notation(x) for x in g) for g in groups)


def group_hits(groups: Groups, text: str) -> tuple[bool, ...]:
    hay = f" {_norm(normalize_notation(text))} "
    return tuple(any(f" {f} " in hay for leaf in g for f in _forms(leaf)) for g in groups)


def score(groups: Groups, answer_text: str) -> dict[str, float]:
    hits = sum(group_hits(groups, answer_text))
    return {"loose": hits / len(groups), "strict": float(hits == len(groups))}


def visible(text: str) -> str:
    """The answer after the model's own thinking block (official `post_think_report` for Qwen:
    text after the first `</think>`; an unclosed draw has no visible answer)."""
    return text.split("</think>", 1)[1].strip() if "</think>" in text else ""


# ---------------------------------------------------------------- message recall

GAP = "\n RCCGAPBOUNDARY \n"  # separates runs, workers and reports; matches no reference


def retained_text(ids: Sequence[int], keep: Sequence[int],
                  decode: Callable[[list[int]], str]) -> str:
    """One sender's kept prompt tokens decoded run by run (never across a dropped position);
    kept positions past the prompt (latent rows) carry no token and are skipped."""
    runs, start, last = [], -2, -2
    for i in keep:
        if i >= len(ids):
            break
        if i != last + 1:
            if start >= 0:
                runs.append(decode(list(ids[start : last + 1])))
            start = i
        last = i
    if start >= 0:
        runs.append(decode(list(ids[start : last + 1])))
    return GAP.join(runs)
