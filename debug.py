"""
test_fixed_model.py
"""

import torch
from src.model import get_model

# 새로운 모델
model = get_model('efficientnet_b4', 17, 0.3)

print("\n" + "="*70)
print("Dropout 레이어 확인")
print("="*70)

dropout_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
        dropout_layers.append((name, module.p))

print(f"\nDropout 레이어: {len(dropout_layers)}개")
for name, p in dropout_layers:
    print(f"  {name}: p={p}")

# Train/Eval 차이 테스트
print("\n" + "="*70)
print("Train/Eval 모드 차이")
print("="*70)

dummy_input = torch.randn(2, 3, 300, 300)

model.train()
with torch.no_grad():
    out1 = model(dummy_input)

model.eval()
with torch.no_grad():
    out2 = model(dummy_input)

diff = (out1 - out2).abs().mean()
print(f"차이: {diff:.6f}")

if diff < 0.05:
    print("✅ 차이 거의 없음")
elif diff < 0.15:
    print("✅ 정상 범위")
elif diff < 0.30:
    print("⚠️  약간 큼")
else:
    print("❌ 너무 큼!")
