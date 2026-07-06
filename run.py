import subprocess
import sys
from pathlib import Path

app = Path(__file__).parent / "app.py"

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(app),
]

print("Executing:", cmd)

subprocess.run(cmd, check=True)