"""
src/transforms.py
Test의 augmentation 대응
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(image_size):
    """
    Test에도 augmentation이 있으므로
    Train에서 강한 증강으로 대응!
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # 기하학적 변환 (강하게!)
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
        ], p=0.5),
        
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        
        # 원근 변환 (문서 스캔 효과)
        A.Perspective(
            scale=(0.05, 0.1),
            p=0.3
        ),
        
        # 색상/밝기 변환 (강하게!)
        A.OneOf([
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
        ], p=0.7),
        
        # 노이즈/블러 (문서 품질 저하)
        A.OneOf([
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                p=1.0
            ),
            A.MotionBlur(
                blur_limit=7,
                p=1.0
            ),
            A.MedianBlur(
                blur_limit=5,
                p=1.0
            ),
        ], p=0.5),
        
        # 그림자/왜곡 효과
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.3
        ),
        
        # 압축 artifacts
        A.ImageCompression(
            quality_lower=75,
            quality_upper=100,
            p=0.3
        ),
        
        # Coarse Dropout (일부 영역 가리기)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
        
        # 정규화
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transform(image_size):
    """Val은 증강 없이"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])