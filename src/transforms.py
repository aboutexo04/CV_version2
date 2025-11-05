"""
src/transforms.py
더 강력한 증강
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size):
    """Train 증강 (강력하게!)"""
    return A.Compose([
        # ✅ 크기 관련 (더 다양하게)
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.7, 1.0),      # 0.8 → 0.7 (더 다양하게)
            ratio=(0.8, 1.2)
        ),
        
        # ✅ 기하학적 변환 (강화)
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),          # 15 → 20
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,                 # 0.1 → 0.2
            rotate_limit=20,                 # 15 → 20
            p=0.5
        ),
        
        # ✅ 색상 변환 (강화)
        A.OneOf([
            A.ColorJitter(
                brightness=0.3,              # 0.2 → 0.3
                contrast=0.3,
                saturation=0.3,
                hue=0.1,                     # 0.05 → 0.1
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
        ], p=0.8),
        
        # ✅ 노이즈/블러 (추가!)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),  # 30% 확률로 노이즈/블러
        
        # ✅ GridDropout 추가 (일부 가리기)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.3
        ),
        
        # Normalize & Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transform(image_size):
    """Validation 증강 (변경 없음)"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])