"""
src/model.py
- 모델만
"""

import timm
import torch.nn as nn

def get_model(model_name, num_classes=17, dropout=0.3):
    """
    모델 생성
    ⚠️ pretrained=True 확인!
    """
    print(f"🔨 모델 생성: {model_name}")
    
    # timm으로 생성
    model = timm.create_model(
        model_name,
        pretrained=True,  # ⚠️ 매우 중요!
        num_classes=num_classes,
        drop_rate=dropout
    )
    
    return model

# ========== 체크용 함수 ==========
def test_model():
    """모델 정상 작동 확인"""
    import torch
    
    model = get_model('efficientnet_b3', num_classes=17, dropout=0.3)
    
    # 더미 입력
    dummy_input = torch.randn(2, 3, 300, 300)
    
    # Forward
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print("✅ 모델 체크:")
    print(f"  입력 shape: {dummy_input.shape}")
    print(f"  출력 shape: {output.shape}")
    print(f"  출력 범위: [{output.min():.3f}, {output.max():.3f}]")
    
    # 예상 결과
    assert output.shape == (2, 17), "출력 shape 오류!"
    
    print("✅ 모델 체크 완료!")

if __name__ == '__main__':
    test_model()