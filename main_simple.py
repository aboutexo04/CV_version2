"""
main_simple.py (완전 수정본 - CUDA 확실히 사용)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

def main():
    """메인 학습 함수"""
    
    from src.config import Config
    from src.transforms import get_train_transform
    from src.dataset import DocumentDataset
    from src.model import get_model
    
    # ========== 설정 ==========
    cfg = Config()
    
    # ✅ cfg.DEVICE 사용! (새로 만들지 말고)
    device = cfg.DEVICE
    
    print("="*70)
    print("🔧 Configuration")
    print("="*70)
    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Device: {device}")  # ← 여기서 cuda 확인!
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌❌❌ CPU 모드입니다!")
        print("CUDA가 안 되는 이유를 확인하세요!")
    
    print(f"Image Size: {cfg.IMAGE_SIZE}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Epochs: {cfg.EPOCHS}")
    print(f"LR: {cfg.LR}")
    print("="*70)
    print()

    # Seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.SEED)

    # ========== 데이터 ==========
    print("📂 데이터 로드...")
    train_df = pd.read_csv(f'{cfg.DATA_DIR}/train.csv')
    print(f"  Train: {len(train_df)}개")
    print(f"  클래스: {train_df['target'].nunique()}개")
    print("\n클래스 분포:")
    print(train_df['target'].value_counts().sort_index())

    train_transform = get_train_transform(cfg.IMAGE_SIZE)
    train_dataset = DocumentDataset(train_df, cfg.TRAIN_DIR, train_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    print(f"\n✅ DataLoader: {len(train_loader)} batches")

    # ========== 모델 ==========
    print("\n🔨 모델 생성...")
    model = get_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.DROPOUT)
    model = model.to(device)  # ← cfg.DEVICE 사용
    
    print(f"✅ 모델이 {device}에 로드됨")
    
    # 모델이 정말 CUDA에 있는지 확인
    print(f"✅ 첫 번째 파라미터 device: {next(model.parameters()).device}")

    # ========== Optimizer & Loss ==========
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n✅ Optimizer: Adam (LR={cfg.LR})")
    print(f"✅ Loss: CrossEntropyLoss")

    # ========== 학습 ==========
    print(f"\n{'='*70}")
    print(f"🔥 학습 시작!")
    print(f"{'='*70}\n")

    best_loss = float('inf')
    best_acc = 0.0

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.EPOCHS}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)  # ← cfg.DEVICE 사용
            labels = labels.to(device)  # ← cfg.DEVICE 사용
            
            # 첫 배치에서 확인
            if epoch == 0 and batch_idx == 0:
                print(f"\n✅ 첫 배치 device 확인:")
                print(f"   Images device: {images.device}")
                print(f"   Labels device: {labels.device}")
                print()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Acc:  {accuracy:.2f}%")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✨ Best 모델 저장!")

    print(f"\n{'='*70}")
    print("🎉 학습 완료!")
    print(f"{'='*70}")
    print(f"\n💾 Best 모델:")
    print(f"   Loss: {best_loss:.4f}")
    print(f"   Acc:  {best_acc:.2f}%")
    print(f"   파일: best_model.pth\n")


if __name__ == '__main__':
    main()