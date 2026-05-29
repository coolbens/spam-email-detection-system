from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "app" / "app.py"
PORT = os.getenv("PORT", "8501")

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(APP_PATH),
    "--server.port",
    PORT,
    "--server.address",
    "0.0.0.0",
]
raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
