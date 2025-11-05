"""
src/model.py (완전 수정)
"""

import timm
import torch.nn as nn


def get_model(model_name, num_classes, dropout):
    """모델 생성 (Dropout 확실히 적용)"""
    print(f"🔨 모델 생성: {model_name}")
    
    # 1. Pretrained 모델 로드 (num_classes 지정 안 함!)
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0  # ← 0으로! (classifier 제거)
    )
    
    # 2. 수동으로 classifier 추가
    num_features = model.num_features  # 또는 model.classifier.in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),  # ← Dropout 명시적 추가!
        nn.Linear(num_features, num_classes)
    )
    
    print(f"   Dropout: {dropout}")
    print(f"   Features: {num_features} → {num_classes}")
    
    return model