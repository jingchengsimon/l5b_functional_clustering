"""Run one fixed cell1 Headley-50 array unit."""

import itertools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


experiment = Path(os.environ["EXPERIMENT"])
phase = os.environ["PHASE"]
index = int(os.environ["SLURM_ARRAY_TASK_ID"])
config = json.loads((experiment / "config.json").read_text())
pairs = list(itertools.product(("basal", "apical"), range(3), range(1, 101)))
section, range_index, epoch = pairs[index]
results_root = Path(config["results_root"])
folder = f"{section}_range{range_index}_{phase}_invivo_{config['channel_suffix']}"
leaf = results_root / folder / "1" / str(epoch)
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


def valid(path):
    try:
        return all((path / name).is_file() and (path / name).stat().st_size for name in required) and np.load(
            path / "soma_v_array.npy", mmap_mode="r"
        ).shape == (40001, 1, 37, 1)
    except Exception:
        return False


if valid(leaf):
    print(f"[VALID] {phase}-{index} {leaf}")
    raise SystemExit(0)

if leaf.exists():
    archive = experiment / "recovery" / "partial" / (
        f"{phase}-{index}-job{os.environ.get('SLURM_JOB_ID', 'manual')}-{time.time_ns()}"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(leaf, archive)
    print(f"[ARCHIVE] {leaf} -> {archive}", flush=True)

command = [
    config["python"],
    "-u",
    "L5b_simulation.py",
    "--results_root", str(results_root),
    "--sec_type", section,
    "--dis_to_root", str(range_index),
    "--start_epoch", str(epoch),
    "--num_epochs", "1",
    *config["cli"],
    "--spat_cond", phase,
]
print("[RUN]", " ".join(command), flush=True)
subprocess.run(command, check=True)
if not valid(leaf):
    raise RuntimeError(f"Incomplete output: {leaf}")
print(f"[COMPLETE] {phase}-{index} {leaf}", flush=True)
