"""
src/config.py (확실하게)
"""

import torch

class Config:
    # 데이터
    DATA_DIR = 'data'
    TRAIN_DIR = 'data/train'
    TEST_DIR = 'data/test'
    
    # 모델
    MODEL_NAME = 'efficientnet_b4'
    NUM_CLASSES = 17
    
    # 학습
    IMAGE_SIZE = 300
    BATCH_SIZE = 24
    EPOCHS = 10
    LR = 0.001
    
    # 정규화
    DROPOUT = 0.3
    WEIGHT_DECAY = 0.0001
    
    # 기타
    SEED = 42
    NUM_WORKERS = 0

    # K-Fold Cross Validation
    USE_KFOLD = True  # K-Fold 사용 여부
    N_FOLDS = 5        # Fold 개수
    LABEL_SMOOTHING = 0.2
    
    # ✅ Device (CUDA, MPS, CPU 자동 선택)
    def __init__(self):
        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            print(f"✅ CUDA 감지됨: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.DEVICE = torch.device('mps')
            print("✅ MPS 감지됨 (Apple Silicon)")
        else:
            self.DEVICE = torch.device('cpu')
            print("⚠️  GPU 없음 - CPU 사용")