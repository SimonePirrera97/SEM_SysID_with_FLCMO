"""Run the Example 3 replicate campaign sequentially and restart-safely."""
from __future__ import annotations

import json
import os
import subprocess
import time
import csv
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CODE_NEW = HERE.parent
WORKSPACE = CODE_NEW.parent
PYTHON = CODE_NEW / ".venv" / "bin" / "python"
STATUS = HERE / "results" / "replicates" / "overnight_campaign_status.json"


def update(stage: str, state: str) -> None:
    if STATUS.exists():
        data = json.loads(STATUS.read_text())
    else:
        data = {"protocol": "four additional initializations for 5 models and 4 methods"}
    data[stage] = {"state": state, "time_utc": datetime.now(timezone.utc).isoformat()}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(data, indent=2) + "\n")


def run(stage: str, command: list[str], env: dict[str, str] | None = None) -> None:
    update(stage, "running")
    subprocess.run(command, cwd=WORKSPACE, env=env, check=True)
    update(stage, "complete")


def wait_for_cmo() -> None:
    """MATLAB is launched separately because it requires native macOS execution."""
    path = HERE / "results" / "replicates" / "cmo_replicates.csv"
    update("cmo_replicates", "running")
    while True:
        if path.exists():
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            keys = {(int(row["replicate"]), int(row["paperRow"])) for row in rows}
            if len(keys) >= 20:
                update("cmo_replicates", "complete")
                return
        time.sleep(15)


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    wait_for_cmo()
    run("adam_replicates", [str(PYTHON), str(HERE / "run_adam_replicates.py")], env)
    run("full_sequence_nnoe10", [str(PYTHON), str(HERE / "run_full_nnoe10_fatigue.py")], env)
    update("campaign", "complete")


if __name__ == "__main__":
    main()
