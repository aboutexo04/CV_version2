"""
src/data_loader.py
"""

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

from .dataset import DocumentDataset
from .transforms import get_train_transform, get_val_transform


def get_dataloaders(cfg):
    """Train/Val DataLoader 생성"""
    
    print("📂 데이터 로드...")
    full_train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    
    # Train/Val Split
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.3,
        random_state=cfg.SEED,
        stratify=full_train_df['target']
    )
    
    print(f"  Train: {len(train_df)}개")
    print(f"  Val: {len(val_df)}개\n")

    # Transform
    train_transform = get_train_transform(cfg.IMAGE_SIZE)
    val_transform = get_val_transform(cfg.IMAGE_SIZE)
    
    # Dataset
    train_dataset = DocumentDataset(train_df, cfg.TRAIN_DIR, train_transform)
    val_dataset = DocumentDataset(val_df, cfg.TRAIN_DIR, val_transform)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(cfg.DEVICE.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(cfg.DEVICE.type == 'cuda')
    )

    return train_loader, val_loader


def get_kfold_dataloaders(cfg, fold):
    """K-Fold를 위한 Train/Val DataLoader 생성

    Args:
        cfg: Config 객체
        fold: 현재 Fold 번호 (0부터 시작)

    Returns:
        train_loader, val_loader
    """
    print(f"📂 데이터 로드 (Fold {fold+1}/{cfg.N_FOLDS})...")
    full_train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')

    # StratifiedKFold 설정
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)

    # 현재 Fold의 train/val 인덱스 가져오기
    train_idx, val_idx = list(skf.split(full_train_df, full_train_df['target']))[fold]

    train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

    print(f"  Train: {len(train_df)}개")
    print(f"  Val: {len(val_df)}개\n")

    # Transform
    train_transform = get_train_transform(cfg.IMAGE_SIZE)
    val_transform = get_val_transform(cfg.IMAGE_SIZE)

    # Dataset
    train_dataset = DocumentDataset(train_df, cfg.TRAIN_DIR, train_transform)
    val_dataset = DocumentDataset(val_df, cfg.TRAIN_DIR, val_transform)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(cfg.DEVICE.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(cfg.DEVICE.type == 'cuda')
    )

    return train_loader, val_loader