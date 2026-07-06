import subprocess
import sys
from pathlib import Path

app = Path(__file__).parent / "app.py"

cmd = [sys.executable, "-m", "streamlit", "run", str(app)]

print("Executing:", " ".join(cmd))

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\nShutting down.")
except subprocess.CalledProcessError as e:
    print(f"Streamlit exited with error code {e.returncode}")
    sys.exit(e.returncode)
