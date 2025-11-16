"""
inference.py
- 추론만 담당
- main과 완전 분리
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.config import Config
from src.transforms import get_val_transform
from src.dataset import DocumentDataset
from src.model import get_model
from torch.utils.data import DataLoader

# ========== 설정 ==========
cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("🔮 추론 시작")
print("="*70)

# ========== 모델 로드 ==========
print("\n📦 모델 로드...")
model = get_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.DROPOUT)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()
print("✅ 모델 로드 완료")

# ========== 테스트 데이터 ==========
print("\n📂 테스트 데이터 로드...")
test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
print(f"  Test: {len(test_df)}개")

# ⚠️ ID 순서 확인!
print("\n⚠️  ID 체크:")
print(f"  첫 ID: {test_df['ID'].iloc[0]}")
print(f"  마지막 ID: {test_df['ID'].iloc[-1]}")

# Dataset
transform = get_val_transform(cfg.IMAGE_SIZE)
test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,  # ⚠️ 절대 섞지 않기!
    num_workers=cfg.NUM_WORKERS
)

# ========== 추론 ==========
print("\n🔮 추론 중...")
predictions = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        pred = outputs.argmax(dim=1).item()
        predictions.append(pred)

# ========== 제출 파일 ==========
test_df['target'] = predictions

print("\n📊 예측 분포:")
print(test_df['target'].value_counts().sort_index())

# 저장
test_df.to_csv('submission.csv', index=False)

print("\n" + "="*70)
print("✅ 제출 파일 생성: submission.csv")
print("="*70)

# ========== 최종 체크 ==========
print("\n🔍 최종 검증:")
sample = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
submission = pd.read_csv('submission.csv')

checks = {
    'Shape 일치': submission.shape == sample.shape,
    'ID 순서 일치': (submission['ID'] == sample['ID']).all(),
    'Target 타입': submission['target'].dtype == 'int64',
    'Target 범위': (submission['target'].min() >= 0) and (submission['target'].max() <= 16),
    '결측치 없음': submission['target'].isnull().sum() == 0,
}

for check, result in checks.items():
    status = "✅" if result else "❌"
    print(f"{status} {check}")

if all(checks.values()):
    print("\n🎉 모든 검증 통과! 제출 가능!")
else:
    print("\n⚠️  문제 발견! 수정 필요!")