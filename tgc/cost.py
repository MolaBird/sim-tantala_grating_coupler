"""FlexCredit / run-time estimates and a confirmation prompt before submitting.

* Cost: each distinct simulation is uploaded once (free) to the tidy3d folder
  ``tgc-estimates``, priced with ``web.estimate_cost`` and deleted again.
  tidy3d reports a *maximum* (full ``run_time``) and usually a *typical* value
  (the field decays and the run stops early). An optimisation iteration is a
  forward + an adjoint simulation; the adjoint is assumed to cost the same as
  the forward one.
* Time: there is no server-side time estimate. Every finished iteration
  stores its wall time (upload + queue + solve + download) per
  cell x time step in ``data-v2/_timings.json`` (shared by all studies), and
  the median of those rates is used. Before any record exists, a rough
  default is used and labelled as such.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import tidy3d as td

from .config import DATA_ROOT

TIMING_FILE = DATA_ROOT / "_timings.json"
DEFAULT_SECONDS_PER_SIM = 240.0  # rough guess, used only until something is measured
ESTIMATE_FOLDER = "tgc-estimates"


def work(sim: td.Simulation) -> float:
    return float(sim.num_cells) * float(sim.num_time_steps)


# ---------------- cost ----------------
def flex_credits(sim: td.Simulation, label: str) -> dict:
    import tidy3d.web as web
    from tidy3d.web.api.task_api import estimate_cost_info, get_info

    task_id = web.upload(sim, task_name=f"estimate_{label}", folder_name=ESTIMATE_FOLDER, verbose=False)
    try:
        est = estimate_cost_info(task_id, verbose=False)
        typical = getattr(est, "typical", None) or get_info(task_id).estFlexUnitTypical
        return dict(max=float(est.maximum), typical=float(typical) if typical else None)
    finally:
        try:
            web.delete(task_id)
        except Exception as e:  # noqa: BLE001 - deletion failure must not block the run
            td.log.warning(f"could not delete estimate task {task_id}: {e}")


# ---------------- time ----------------
def _records() -> list:
    if TIMING_FILE.exists():
        return json.loads(TIMING_FILE.read_text())
    return []


def record_timing(sim: td.Simulation, seconds_per_sim: float, study: str) -> None:
    rec = _records()
    rec.append(dict(study=study, work=work(sim), seconds=float(seconds_per_sim)))
    TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMING_FILE.write_text(json.dumps(rec[-200:], indent=1))


def seconds_per_sim(sim: td.Simulation) -> tuple[float, str]:
    rec = _records()
    if not rec:
        return DEFAULT_SECONDS_PER_SIM, "rough default (nothing measured yet)"
    rate = float(np.median([r["seconds"] / r["work"] for r in rec]))
    return rate * work(sim), f"from {len(rec)} measured simulation(s)"


# ---------------- report + prompt ----------------
def _fmt_time(s: float) -> str:
    h, m = divmod(int(round(s / 60)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def summarize(rows: list[dict]) -> str:
    """rows: dict(label, n_sims, sim, credits=dict(max, typical))."""
    lines = [f"{'item':<34}{'sims':>5}{'max FC':>10}{'typical FC':>12}{'time':>14}"]
    tot = dict(max=0.0, typical=0.0, time=0.0)
    typical_ok = True
    notes = set()
    for r in rows:
        n, c = r["n_sims"], r["credits"]
        sec, note = seconds_per_sim(r["sim"])
        notes.add(note)
        typ = c["typical"]
        typical_ok &= typ is not None
        tot["max"] += n * c["max"]
        tot["typical"] += n * (typ if typ is not None else c["max"])
        tot["time"] += n * sec
        typ_s = f"{n * typ:.2f}" if typ is not None else "n/a"
        lines.append(f"{r['label']:<34}{n:>5}{n * c['max']:>10.2f}{typ_s:>12}{_fmt_time(n * sec):>14}")
    lines.append("-" * len(lines[0]))
    typ_tot = f"{tot['typical']:.2f}" if typical_ok else f"<={tot['typical']:.2f}"
    lines.append(f"{'total':<34}{'':>5}{tot['max']:>10.2f}{typ_tot:>12}{_fmt_time(tot['time']):>14}")
    lines.append(f"time estimate: {'; '.join(sorted(notes))}")
    return "\n".join(lines)


def confirm(text: str, assume_yes: bool = False) -> bool:
    print(text)
    if assume_yes:
        print("--yes given: submitting without asking.")
        return True
    try:
        ans = input("Submit to Tidy3D? [y/N] ").strip().lower()
    except EOFError:  # non-interactive: never submit silently
        print("No interactive input available; not submitting (use --yes).", file=sys.stderr)
        return False
    return ans in ("y", "yes")
