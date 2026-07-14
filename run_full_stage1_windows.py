import os
import runpy
import sys
from pathlib import Path

os.environ["SLTR_DATASET"] = "rocstories"
os.environ["TRAIN_SIZE"] = "1000000"
os.environ["TRAIN_BATCH_SIZE"] = "2"
os.environ["STAGE1_EPOCHS"] = "3"
os.environ["LATENT_DIM"] = "768"
os.environ["MAX_SEQ_LEN"] = "128"
os.environ["PROMPT_LEN"] = "64"
os.environ["STAGE1_COSMOS_LIKE"] = "true"
os.environ["STAGE1_VARIANT"] = "cosmos"
os.environ["STAGE1_CHECKPOINT"] = "stage1_rocstories_768_cosmos_best.pt"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import datasets

datasets.config.TORCHVISION_AVAILABLE = False

import torch.utils.data as torch_data

OriginalDataLoader = torch_data.DataLoader


def WindowsDataLoader(*args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False
    return OriginalDataLoader(*args, **kwargs)


torch_data.DataLoader = WindowsDataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

runpy.run_path(
    str(PROJECT_ROOT / "src" / "parallel_decoder.py"),
    run_name="__main__",
)
