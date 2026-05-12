"""Train the Stage 1 latent autoencoder.

This is a thin entrypoint around ``src/parallel_decoder.py``, which still owns
the current Stage 1 model and training loop.
"""

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

runpy.run_path(str(PROJECT_ROOT / "src" / "parallel_decoder.py"), run_name="__main__")
