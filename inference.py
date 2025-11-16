# 1. 새 파일 생성

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import json
from datetime import datetime

def main(exp_dir=None):
    from src.config import Config
    from src.transforms import get_val_transform
    from src.dataset import DocumentDataset
    from src.model import get_model
    from torch.utils.data import DataLoader

    cfg = Config()
    device = cfg.DEVICE

    if exp_dir is None:
        exp_dirs = sorted(Path('experiments').glob('exp_*'))
        if not exp_dirs:
            raise ValueError("실험 폴더가 없습니다!")
        exp_dir = exp_dirs[-1]
        print(f"📁 가장 최근 실험 사용: {exp_dir}")
    else:
        exp_dir = Path(exp_dir)

    # 실험의 설정 로드 (results.json에서)
    results_files = list(exp_dir.glob('results_fold*.json'))
    avg_val_f1 = None

    if results_files:
        # 모든 fold의 결과 로드
        fold_results = []
        for result_file in sorted(results_files):
            with open(result_file, 'r') as f:
                fold_data = json.load(f)
                fold_results.append(fold_data)

        exp_config = fold_results[0]['config']
        model_name = exp_config['model_name']
        image_size = exp_config['image_size']
        dropout = exp_config['dropout']

        # 평균 val F1 계산
        val_f1_scores = [fold['best_results']['val_f1'] for fold in fold_results]
        avg_val_f1 = np.mean(val_f1_scores)

        print(f"📋 실험 설정 로드: {model_name}, img_size={image_size}, dropout={dropout}")
        print(f"📊 평균 Validation F1: {avg_val_f1:.4f}")
    else:
        # 설정 파일이 없으면 현재 config 사용
        model_name = cfg.MODEL_NAME
        image_size = cfg.IMAGE_SIZE
        dropout = cfg.DROPOUT
        print(f"⚠️  실험 설정 없음 - 현재 config 사용")
    
    # K-Fold 모델 확인 (두 가지 구조 지원)
    folds_dir = exp_dir / 'folds'
    fold_models = list(exp_dir.glob('best_model_fold*.pth'))

    is_kfold = folds_dir.exists() or len(fold_models) > 0

    print("="*70)
    print(f"🔮 추론 시작")
    print(f"   실험: {exp_dir.name}")
    if is_kfold:
        if folds_dir.exists():
            fold_dirs = sorted(folds_dir.glob('fold*'))
            print(f"   모드: K-Fold 앙상블 ({len(fold_dirs)} Folds)")
        else:
            print(f"   모드: K-Fold 앙상블 ({len(fold_models)} Folds)")
    else:
        print(f"   모드: Single Model")
    print("="*70)
    print()

    print("📂 테스트 데이터 로드...")
    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    print(f"  Test: {len(test_df)}개\n")

    transform = get_val_transform(image_size)
    test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    if is_kfold:
        all_predictions = []

        # folds/ 디렉토리 구조 또는 직접 fold 모델 파일 지원
        if folds_dir.exists():
            fold_dirs = sorted(folds_dir.glob('fold*'))
            model_paths = [fold_dir / 'model.pth' for fold_dir in fold_dirs]
        else:
            model_paths = sorted(exp_dir.glob('best_model_fold*.pth'))

        for fold_idx, model_path in enumerate(model_paths):
            print(f"📦 Fold {fold_idx+1}/{len(model_paths)} 모델 로드...")
            print(f"   {model_path.name}")

            model = get_model(model_name, cfg.NUM_CLASSES, dropout)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model = model.to(device)
            model.eval()

            print(f"🔮 Fold {fold_idx+1} 추론 중...")
            fold_preds = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'Fold {fold_idx+1}', leave=False):
                    if isinstance(batch, (tuple, list)):
                        images = batch[0]
                    else:
                        images = batch

                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    fold_preds.append(probs.cpu().numpy())

            fold_preds = np.concatenate(fold_preds, axis=0)
            all_predictions.append(fold_preds)

            print(f"✅ Fold {fold_idx+1} 완료\n")

        print("🔀 앙상블 중...")
        ensemble_probs = np.mean(all_predictions, axis=0)
        predictions = np.argmax(ensemble_probs, axis=1).tolist()
        
    else:
        model_path = exp_dir / 'best_model.pth'
        if not model_path.exists():
            raise ValueError(f"모델 파일이 없습니다: {model_path}")

        print("📦 모델 로드...")
        model = get_model(model_name, cfg.NUM_CLASSES, dropout)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model = model.to(device)
        model.eval()
        print("✅ 모델 로드 완료\n")

        print("🔮 추론 중...")
        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Inference'):
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                predictions.extend(preds.cpu().numpy().tolist())

    test_df['target'] = predictions

    # 제출 파일명 생성 (날짜_시간_모델명_valF1.csv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = [timestamp]

    if is_kfold:
        filename_parts.append(f"ensemble_{len(model_paths)}fold")
    else:
        filename_parts.append("single")

    filename_parts.append(model_name)

    if avg_val_f1 is not None:
        filename_parts.append(f"valF1_{avg_val_f1:.4f}")

    submission_filename = "_".join(filename_parts) + ".csv"
    submission_path = exp_dir / submission_filename
    test_df.to_csv(submission_path, index=False)

    print("\n📊 예측 분포:")
    print(test_df['target'].value_counts().sort_index())

    print("\n" + "="*70)
    print(f"✅ 제출 파일: {submission_path}")
    print("="*70)

    print("\n🔍 최종 검증:")
    sample = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')

    checks = {
        'Shape 일치': test_df.shape == sample.shape,
        'ID 순서 일치': (test_df['ID'] == sample['ID']).all(),
        'Target 타입': test_df['target'].dtype in ['int64', 'int32'],
        'Target 범위': (test_df['target'].min() >= 0) and (test_df['target'].max() <= 16),
        '결측치 없음': test_df['target'].isnull().sum() == 0,
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
    parser.add_argument('--exp_dir', type=str, default=None)
    args = parser.parse_args()
    
    main(args.exp_dir)


# 2. 실행!
