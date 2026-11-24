#!/usr/bin/env python3
"""Sequential job runner for the CR arms (one subprocess per run, skip finished runs).

Usage:
    python scripts/run_jobs.py B:42:0 C:42:0.2 ...
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))

import paths  # noqa: E402

PY = sys.executable
STATUS = paths.REPORTS / "jobs.jsonl"


def main() -> int:
    specs = sys.argv[1:]
    if not specs:
        print("usage: run_jobs.py ARM:SEED:BETA [...]")
        return 2
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    log = STATUS.open("a", encoding="utf-8")
    for spec in specs:
        arm, seed, beta = spec.split(":")
        seed, beta = int(seed), float(beta)
        tag = f"{arm}_s{seed}" + (f"_b{beta:g}" if arm in ("C", "D") else "")
        last = paths.RUNS / "arms" / tag / "weights" / "last.pt"
        if last.is_file() and last.stat().st_size > 0:
            print(f"[skip] {tag} already has last.pt", flush=True)
            log.write(json.dumps({"tag": tag, "status": "skipped",
                                  "at": datetime.now().isoformat(timespec="seconds")}) + "\n")
            continue
        t0 = time.time()
        cmd = [PY, str(TASK / "scripts" / "train_arm.py"), "--arm", arm, "--seed", str(seed),
               "--beta", str(beta), "--tag", tag]
        print(f"[run] {tag}", flush=True)
        r = subprocess.run(cmd, cwd=str(TASK), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        (paths.REPORTS / f"_{tag}.log").write_text(
            (r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""), encoding="utf-8")
        row = {"tag": tag, "arm": arm, "seed": seed, "beta": beta, "returncode": r.returncode,
               "elapsed_s": round(time.time() - t0, 1),
               "at": datetime.now().isoformat(timespec="seconds"),
               "status": "ok" if r.returncode == 0 else "failed"}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"[{row['status']}] {tag} {row['elapsed_s']}s", flush=True)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
