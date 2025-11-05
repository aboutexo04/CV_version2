"""
src/data_loader.py
StratifiedGroupKFold (클래스 가중치 없음)
"""

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from .dataset import DocumentDataset
from .transforms import get_train_transform, get_val_transform


def _extract_doc_id_column(df: pd.DataFrame):
    """이미지 ID에서 문서 ID 추출"""
    if 'ID' not in df.columns:
        raise ValueError("❌ train.csv에 'ID' 열이 없습니다!")
    
    # ID에서 확장자 제거하여 doc_id 생성
    # 예: '002f99746285dfdd.jpg' → '002f99746285dfdd'
    df['doc_id'] = df['ID'].apply(lambda x: str(x).split('.')[0])
    return df


def get_dataloaders(cfg):
    """단일 Train/Val DataLoader 생성"""
    
    print("📂 데이터 로드...")
    full_train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    
    # 문서 ID 추출
    full_train_df = _extract_doc_id_column(full_train_df)
    
    # StratifiedGroupKFold (첫 번째 fold만 사용)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    
    splits = list(sgkf.split(
        full_train_df, 
        full_train_df['target'], 
        groups=full_train_df['doc_id']
    ))
    
    train_idx, val_idx = splits[0]
    
    train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_train_df.iloc[val_idx].reset_index(drop=True)
    
    print(f"  Train: {len(train_df)}개 ({len(train_df['doc_id'].unique())}개 문서)")
    print(f"  Val: {len(val_df)}개 ({len(val_df['doc_id'].unique())}개 문서)\n")
    
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
    """K-Fold DataLoader 생성 (StratifiedGroupKFold)"""
    
    print(f"📂 데이터 로드... (Fold {fold+1}/{cfg.N_FOLDS})")
    full_train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    
    # 문서 ID 추출
    full_train_df = _extract_doc_id_column(full_train_df)
    
    # StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.N_FOLDS, 
        shuffle=True, 
        random_state=cfg.SEED
    )
    
    splits = list(sgkf.split(
        full_train_df,
        full_train_df['target'],
        groups=full_train_df['doc_id']
    ))
    
    train_idx, val_idx = splits[fold]
    
    train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_train_df.iloc[val_idx].reset_index(drop=True)
    
    print(f"  Train: {len(train_df)}개 ({len(train_df['doc_id'].unique())}개 문서)")
    print(f"  Val: {len(val_df)}개 ({len(val_df['doc_id'].unique())}개 문서)\n")
    
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