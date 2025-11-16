"""
inference_calibrated.py (수정)
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import argparse

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

    # folds/ 디렉토리 구조 또는 직접 fold 모델 파일 지원
    if folds_dir.exists():
        model_paths = [fold_dir / 'model.pth' for fold_dir in sorted(folds_dir.glob('fold*'))]
    else:
        model_paths = sorted(exp_dir.glob('best_model_fold*.pth'))

    if len(model_paths) == 0:
        raise ValueError("모델 파일을 찾을 수 없습니다!")

    print("="*70)
    print("🔮 캘리브레이션 추론")
    print("="*70)
    print(f"실험: {exp_dir.name}")
    print(f"Folds: {len(model_paths)}개\n")

    # 데이터 로드
    test_df = pd.read_csv(f'{cfg.DATA_DIR}/sample_submission.csv')
    print(f"Test: {len(test_df)}개")

    transform = get_val_transform(image_size)
    test_dataset = DocumentDataset(test_df, cfg.TEST_DIR, transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Train 분포
    train = pd.read_csv('data/train.csv')
    train_dist = train['target'].value_counts().sort_index().values
    train_dist = train_dist / train_dist.sum()
    
    print("\n📊 Train 분포:")
    for cls in range(17):
        print(f"  클래스 {cls:2d}: {train_dist[cls]*100:5.2f}%")
    
    # Fold별 추론
    all_predictions = []

    for fold_idx, model_path in enumerate(model_paths):
        print(f"\n📦 Fold {fold_idx+1}/{len(model_paths)} 로드...")
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
        print(f"✅ Fold {fold_idx+1} 완료 (shape: {fold_preds.shape})")
    
    # 앙상블
    print("\n🔀 앙상블 중...")
    ensemble_probs = np.mean(all_predictions, axis=0)
    print(f"앙상블 shape: {ensemble_probs.shape}")
    
    # 원본 예측
    raw_preds = ensemble_probs.argmax(axis=1)
    pred_dist_raw = np.bincount(raw_preds, minlength=17) / len(raw_preds)
    
    print("\n📊 원본 예측 분포:")
    for cls in range(17):
        print(f"  클래스 {cls:2d}: {pred_dist_raw[cls]*100:5.2f}%")
    
    # 캘리브레이션
    print("\n⚙️  캘리브레이션 적용...")
    
    # 조정 계수
    calibration = train_dist / (pred_dist_raw + 1e-8)
    calibration = np.clip(calibration, 0.5, 2.0)
    
    print("\n📈 조정 계수:")
    for cls in range(17):
        if calibration[cls] > 1.1 or calibration[cls] < 0.9:
            print(f"  클래스 {cls:2d}: {calibration[cls]:.2f}x")
    
    # 확률 조정
    calibrated_probs = ensemble_probs * calibration
    calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
    
    predictions = calibrated_probs.argmax(axis=1).tolist()

    # 결과 저장 (파일명에 날짜, 시간, F1 스코어 포함)
    test_df['target'] = predictions

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = [timestamp, f"calibrated_ensemble_{len(model_paths)}fold", model_name]

    if avg_val_f1 is not None:
        filename_parts.append(f"valF1_{avg_val_f1:.4f}")

    submission_filename = "_".join(filename_parts) + ".csv"
    submission_path = exp_dir / submission_filename
    test_df.to_csv(submission_path, index=False)

    print(f"\n✅ 저장: {submission_path}")
    
    # 최종 분포
    pred_dist_final = np.bincount(predictions, minlength=17) / len(predictions)
    
    print("\n📊 최종 비교:")
    print(f"{'클래스':<8} {'Train':<10} {'원본':<10} {'조정후':<10}")
    print("-" * 40)
    for cls in range(17):
        print(f"{cls:2d}       {train_dist[cls]*100:5.2f}%    {pred_dist_raw[cls]*100:5.2f}%    {pred_dist_final[cls]*100:5.2f}%")
    
    print("\n" + "="*70)
    print("✅ 완료! 이 파일을 제출하세요:")
    print(f"   {submission_path}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default=None)
    args = parser.parse_args()

    main(args.exp_dir)