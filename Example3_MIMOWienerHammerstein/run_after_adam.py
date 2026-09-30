"""Wait for the restart-safe Adam campaign, then run the full-sequence fit."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CODE_NEW = HERE.parent
PYTHON = CODE_NEW / ".venv" / "bin" / "python"
RESULTS = HERE / "results" / "replicates"
ADAM = RESULTS / "adam_replicates.csv"
STATUS = RESULTS / "overnight_campaign_status.json"


def status(stage: str, state: str) -> None:
    data = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    data[stage] = {"state": state, "time_utc": datetime.now(timezone.utc).isoformat()}
    STATUS.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    status("adam_replicates", "running")
    while True:
        if ADAM.exists():
            with ADAM.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            keys = {(row["method"], int(row["replicate"]), int(row["paper_row"])) for row in rows}
            if len(keys) >= 60:
                break
        time.sleep(15)
    status("adam_replicates", "complete")
    status("full_sequence_nnoe10", "running")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    subprocess.run([str(PYTHON), str(HERE / "run_full_nnoe10_fatigue.py")],
                   cwd=CODE_NEW.parent, env=env, check=True)
    status("full_sequence_nnoe10", "complete")
    status("campaign", "complete")


if __name__ == "__main__":
    main()
