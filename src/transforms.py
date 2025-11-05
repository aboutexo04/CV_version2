"""
src/transforms.py
문서 이미지 분류용 완화된 증강 버전
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size):
    """문서 이미지용 완화된 증강"""
    return A.Compose([
        # 크기: 거의 원본 유지 (문서의 형태 보존)
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.9, 1.0),
            ratio=(0.9, 1.1)
        ),

        # 기하 변환: 약하게 (문서 왜곡 방지)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            p=0.5
        ),

        # 색상 밝기/대비: 자연스러운 범위 내에서만
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),

        # 약한 노이즈 (글자 유지)
        A.GaussNoise(var_limit=(5, 20), p=0.2),

        # 표준 Normalize 및 Tensor 변환
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transform(image_size):
    """Validation/Test 변환 (증강 없음)"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])