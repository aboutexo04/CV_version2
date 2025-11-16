"""
src/utils.py
"""

import torch
import numpy as np
import random
from pathlib import Path


def set_seed(seed=42):
    """재현성을 위한 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment(timestamp):
    """실험 폴더 생성"""
    exp_dir = Path(f'experiments/exp_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir