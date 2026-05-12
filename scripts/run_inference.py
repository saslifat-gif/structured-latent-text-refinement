"""Run conditional Stage 2 inference."""

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

runpy.run_path(
    str(PROJECT_ROOT / "src" / "inference_stage2_conditional.py"),
    run_name="__main__",
)
