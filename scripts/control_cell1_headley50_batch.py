"""Submit, audit, and selectively retry the two-stage cell1 batch."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


experiment = Path(os.environ["EXPERIMENT"])
config = json.loads((experiment / "config.json").read_text())
results_root = Path(config["results_root"])
worker = experiment / "worker.sbatch"
controller = experiment / "controller.sbatch"
required = (
    "soma_v_array.npy",
    "apic_v_array.npy",
    "apic_ica_array.npy",
    "trunk_v_array.npy",
    "basal_v_array.npy",
    "tuft_v_array.npy",
    "section_synapse_df.csv",
    "simulation_params.json",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def leaf(index, phase):
    epoch = index % 100 + 1
    range_index = index // 100 % 3
    section = ("basal", "apical")[index // 300]
    folder = f"{section}_range{range_index}_{phase}_invivo_{config['channel_suffix']}"
    return results_root / folder / "1" / str(epoch)


def valid(path):
    try:
        return all((path / name).is_file() and (path / name).stat().st_size for name in required) and np.load(
            path / "soma_v_array.npy", mmap_mode="r"
        ).shape == (40001, 1, 37, 1)
    except Exception:
        return False


def compress(indices):
    ranges = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            ranges.append(f"{start}-{previous}" if start != previous else str(start))
            start = value
        previous = value
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def sbatch(*args):
    max_attempts = int(config.get("sbatch_max_attempts", 20))
    retry_seconds = float(config.get("sbatch_retry_seconds", 30))
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(["sbatch", "--parsable", *args], text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip().split(";", 1)[0]

        error = (result.stderr or result.stdout).strip()
        qos_limited = "QOSMaxSubmitJobPerUserLimit" in error
        if not qos_limited or attempt == max_attempts:
            raise RuntimeError(f"sbatch failed after {attempt} attempt(s): {error}")
        print(
            f"[retry] sbatch hit QOSMaxSubmitJobPerUserLimit; "
            f"attempt={attempt}/{max_attempts}, sleeping={retry_seconds:g}s",
            flush=True,
        )
        time.sleep(retry_seconds)

    raise AssertionError("unreachable")


def load_state():
    path = experiment / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"created_at": now(), "waves": [], "status": "starting"}


def save_state(state):
    temporary = experiment / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(experiment / "state.json")


def submit_wave(phase, indices, state):
    attempt = 1 + sum(wave["phase"] == phase for wave in state["waves"])
    if attempt > int(config["max_waves_per_phase"]):
        blocked = {"blocked_at": now(), "phase": phase, "missing": indices, "state": state}
        (experiment / "BLOCKED.json").write_text(json.dumps(blocked, indent=2) + "\n")
        raise RuntimeError(f"{phase} still has {len(indices)} missing units after {attempt - 1} waves")

    submitted_indices = indices[:int(config["max_units_per_wave"])]
    array_job = sbatch(
        f"--array={compress(submitted_indices)}%{config['max_concurrency']}",
        f"--job-name=c1h50-{phase}-w{attempt}",
        f"--output={experiment}/logs/%A_%a.out",
        f"--error={experiment}/logs/%A_%a.err",
        f"--export=ALL,EXPERIMENT={experiment},PHASE={phase}",
        str(worker),
    )
    wave = {
        "phase": phase,
        "attempt": attempt,
        "submitted_at": now(),
        "units": len(submitted_indices),
        "array_job_id": array_job,
        "audit_job_id": None,
    }
    state["waves"].append(wave)
    state["status"] = f"awaiting_audit_{phase}"
    save_state(state)

    audit_job = sbatch(
        f"--dependency=afterany:{array_job}",
        f"--job-name=c1h50-audit-{phase}-w{attempt}",
        f"--output={experiment}/logs/%j-controller.out",
        f"--error={experiment}/logs/%j-controller.err",
        f"--export=ALL,EXPERIMENT={experiment},PHASE={phase}",
        str(controller),
        "audit",
    )
    wave["audit_job_id"] = audit_job
    state["status"] = f"running_{phase}"
    save_state(state)
    print(json.dumps(wave, indent=2))


mode = sys.argv[1]
state = load_state()
experiment.joinpath("logs").mkdir(parents=True, exist_ok=True)
if mode == "launch":
    if state["waves"]:
        raise RuntimeError("Batch already launched")
    submit_wave("clus", list(range(600)), state)
elif mode == "audit":
    phase = os.environ["PHASE"]
    missing = [index for index in range(600) if not valid(leaf(index, phase))]
    report = {"checked_at": now(), "phase": phase, "complete": 600 - len(missing), "missing": missing}
    (experiment / f"audit-{phase}.json").write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        submit_wave(phase, missing, state)
    elif phase == "clus":
        submit_wave("distr", list(range(600)), state)
    else:
        state["status"] = "complete"
        state["completed_at"] = now()
        save_state(state)
        (experiment / "COMPLETE").write_text(json.dumps(state, indent=2) + "\n")
        print("[COMPLETE] clus=600 distr=600")
else:
    raise ValueError(f"Unknown mode: {mode}")
