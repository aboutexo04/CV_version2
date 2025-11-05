"""
src/dataset.py
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class DocumentDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['ID']
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        if 'target' in row:
            label = int(row['target'])
            return img, label
        
        return img

# ========== 체크용 함수 ==========
def test_dataset():
    """Dataset 정상 작동 확인"""
    import sys
    from pathlib import Path
    
    # ✅ 프로젝트 루트를 경로에 추가
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 이제 import 가능
    import pandas as pd
    from src.transforms import get_train_transform
    
    # 데이터 로드
    df = pd.read_csv('data/train.csv')
    print(f"✅ CSV 로드: {len(df)}개")
    
    # Dataset 생성
    transform = get_train_transform(300)
    dataset = DocumentDataset(df, 'data/train', transform)
    
    # 첫 샘플 체크
    img, label = dataset[0]
    
    print("✅ Dataset 체크:")
    print(f"  이미지 shape: {img.shape}")
    print(f"  이미지 dtype: {img.dtype}")
    print(f"  라벨: {label} (타입: {type(label)})")
    
    # 여러 샘플 체크
    for i in range(min(5, len(dataset))):
        img, label = dataset[i]
        assert img.shape == (3, 300, 300), f"Sample {i} shape 오류!"
        assert 0 <= label < 17, f"Sample {i} label 범위 오류!"
    
    print(f"✅ {min(5, len(dataset))}개 샘플 체크 완료!")

if __name__ == '__main__':
    test_dataset()