"""Common environment setup: HF cache, random seeds, GPU config."""
import os
import random
import numpy as np
import torch


def setup_hf_cache(cache_dir: str | None = None):
    """Configure Hugging Face cache directory.

    Priority: *cache_dir* arg > ``HF_HOME`` env var > ``~/.cache/huggingface``.
    """
    _hf_cache = cache_dir or os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    os.makedirs(_hf_cache, exist_ok=True)
    os.environ['HF_HOME'] = _hf_cache
    os.environ['HF_HUB_CACHE'] = os.path.join(_hf_cache, 'hub')


def set_seed(seed: int, deterministic: bool = True):
    """Set random seed. Use deterministic=False for faster training (non-reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def setup_single_gpu(gpu_id: int = 0):
    """Configure environment for single-GPU execution. Must be called before importing torch."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
