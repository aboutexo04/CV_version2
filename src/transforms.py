"""
src/transforms.py
- 하나하나 체크하며 작성
- 의심스러운 거 전부 제거
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(image_size):
    """
    학습용 Transform
    ⚠️ 최소한의 증강만!
    """
    return A.Compose([
        # 1. 크기 조정 (확인!)
        A.Resize(image_size, image_size),
        
        # 2. 간단한 증강 하나만
        A.HorizontalFlip(p=0.5),
        
        # 3. 정규화 (ImageNet 값 확인!)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # 4. Tensor 변환
        ToTensorV2(),
    ])

def get_val_transform(image_size):
    """
    검증/테스트용 Transform
    ⚠️ 증강 없음!
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])

# ========== 체크용 함수 ==========
def test_transform():
    """Transform 정상 작동 확인"""
    import numpy as np
    from PIL import Image
    
    # 더미 이미지
    img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    # Train transform
    train_tf = get_train_transform(300)
    result = train_tf(image=img)['image']
    
    print("✅ Transform 체크:")
    print(f"  입력 shape: {img.shape}")
    print(f"  출력 shape: {result.shape}")
    print(f"  출력 dtype: {result.dtype}")
    print(f"  출력 범위: [{result.min():.3f}, {result.max():.3f}]")
    
    # 예상 결과
    assert result.shape == (3, 300, 300), "Shape 오류!"
    assert -3 < result.min() < 0, "Normalize 오류!"
    assert 2 < result.max() < 3, "Normalize 오류!"
    
    print("✅ 모든 체크 통과!")

if __name__ == '__main__':
    test_transform()