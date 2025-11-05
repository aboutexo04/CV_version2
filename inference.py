"""
inference.py (완전 수정)
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def main(exp_dir=None):
    from src.config import Config
    from src.transforms import get_val_transform
    from src.dataset import DocumentDataset
    from src.model import get_model
    from torch.utils.data import DataLoader
    
    cfg = Config()
    device = cfg.DEVICE
    
    # ========== 실험 폴더 확인 ==========
    if exp_dir is None:
        exp_dirs = sorted(Path('experiments').glob('exp_*'))
        if not exp_dirs:
            raise ValueError("실험 폴더가 없습니다! 먼저 학습을 실행하세요.")
        exp_dir = exp_dirs[-1]
        print(f"📁 가장 최근 실험 사용: {exp_dir}")
    else:
        exp_dir = Path(exp_dir)
    
    model_path = exp_dir / 'best_model.pth'
    if not model_path.exists():
        raise ValueError(f"모델 파일이 없습니다: {model_path}")
    
    print("="*70)
    print(f"🔮 추론 시작")
    print(f"   실험: {exp_dir.name}")
    print(f"   모델: {model_path.name}")
    print("="*70)

    # ========== 모델 로드 ==========
    print("\n📦 모델 로드...")
    model = get_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.DROPOUT)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ 모델 로드 완료\n")

    # ========== 테스트 데이터 ==========
    print("📂 테스트 데이터 로드...")
    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    print(f"  Test: {len(test_df)}개\n")

    transform = get_val_transform(cfg.IMAGE_SIZE)
    test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, transform)
    
    # ✅ batch_size=1이 아니라 cfg.BATCH_SIZE 사용
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # 1 → BATCH_SIZE
        shuffle=False,
        num_workers=0
    )

    # ========== 추론 ==========
    print("🔮 추론 중...")
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            # ✅ batch가 tuple인지 tensor인지 확인
            if isinstance(batch, (tuple, list)):
                images = batch[0]  # (images, labels)에서 images만
            else:
                images = batch  # 그냥 images
            
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy().tolist())

    # ========== 제출 파일 저장 ==========
    test_df['target'] = predictions
    
    submission_path = exp_dir / 'submission.csv'
    test_df.to_csv(submission_path, index=False)

    print("\n📊 예측 분포:")
    print(test_df['target'].value_counts().sort_index())

    print("\n" + "="*70)
    print(f"✅ 제출 파일 저장: {submission_path}")
    print("="*70)

    # ========== 최종 검증 ==========
    print("\n🔍 최종 검증:")
    sample = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    submission = test_df

    checks = {
        'Shape 일치': submission.shape == sample.shape,
        'ID 순서 일치': (submission['ID'] == sample['ID']).all(),
        'Target 타입': submission['target'].dtype in ['int64', 'int32'],
        'Target 범위': (submission['target'].min() >= 0) and (submission['target'].max() <= 16),
        '결측치 없음': submission['target'].isnull().sum() == 0,
    }

    all_passed = True
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 모든 검증 통과! 제출 가능!")
    else:
        print("\n⚠️  문제 발견! 수정 필요!")
    
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default=None, 
                       help='실험 폴더 경로 (없으면 최신 사용)')
    args = parser.parse_args()
    
    main(args.exp_dir)