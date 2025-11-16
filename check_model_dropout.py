"""
check_model_dropout.py
"""

import torch
from src.model import get_model

model = get_model('efficientnet_b4', 17, 0.3)

print("="*70)
print("모델 내 Dropout 레이어 확인")
print("="*70)

dropout_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        dropout_layers.append((name, module.p))

print(f"\nDropout 레이어 개수: {len(dropout_layers)}개")
print("\nDropout 레이어 목록:")
for name, p in dropout_layers:
    print(f"  {name}: p={p}")

if len(dropout_layers) == 0:
    print("\n❌ Dropout 레이어가 없습니다!")
elif len(dropout_layers) > 5:
    print(f"\n⚠️  Dropout 레이어가 너무 많습니다! ({len(dropout_layers)}개)")
else:
    print(f"\n✅ Dropout 레이어 개수 정상")

# p 값 확인
if dropout_layers:
    max_p = max(p for _, p in dropout_layers)
    print(f"\n최대 Dropout 비율: {max_p}")
    if max_p > 0.5:
        print(f"⚠️  Dropout 비율이 너무 높습니다!")