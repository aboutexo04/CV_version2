"""
main.py (최종 깔끔 버전)
"""

from datetime import datetime
import numpy as np

def main():
    from src.config import Config
    from src.data_loader import get_dataloaders, get_kfold_dataloaders
    from src.model import get_model
    from src.trainer import Trainer
    from src.utils import setup_experiment, set_seed

    # ========== 실험 설정 ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = setup_experiment(timestamp)

    cfg = Config()

    cfg.MODEL_NAME = 'efficientnet_b4'
    cfg.EPOCHS = 30
    cfg.LR = 0.0001                    # 더 천천히
    cfg.DROPOUT = 0.4                   # 더 강하게
    cfg.WEIGHT_DECAY = 0.001           # 더 강하게
    cfg.IMAGE_SIZE = 300                # 더 작게


    print("="*70)
    print(f"📁 실험: {exp_dir}")
    print(f"🔧 모델: {cfg.MODEL_NAME}")
    print(f"🖥️  Device: {cfg.DEVICE}")
    if cfg.USE_KFOLD:
        print(f"📊 K-Fold: {cfg.N_FOLDS} folds")
    print("="*70)
    print()

    set_seed(cfg.SEED)

    # ========== K-Fold 사용 시 ==========
    if cfg.USE_KFOLD:
        fold_scores = []

        for fold in range(cfg.N_FOLDS):
            print("\n" + "="*70)
            print(f"🔄 Fold {fold+1}/{cfg.N_FOLDS} 시작")
            print("="*70)

            # 데이터 로드
            train_loader, val_loader = get_kfold_dataloaders(cfg, fold)

            # 모델 생성 (각 Fold마다 새로 생성)
            print("🔨 모델 생성...")
            model = get_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.DROPOUT)
            model = model.to(cfg.DEVICE)
            print("✅ 모델 준비 완료\n")

            # 학습
            trainer = Trainer(model, cfg, exp_dir, fold=fold)
            trainer.train(train_loader, val_loader)

            fold_scores.append(trainer.best_val_f1)

            print(f"\n✅ Fold {fold+1} 완료 - Best Val F1: {trainer.best_val_f1:.4f}")

        # 전체 결과 출력
        print("\n" + "="*70)
        print("🎉 K-Fold Cross Validation 완료!")
        print("="*70)
        print("\n📊 각 Fold 결과:")
        for i, score in enumerate(fold_scores):
            print(f"   Fold {i+1}: F1 = {score:.4f}")
        print(f"\n📈 평균 F1: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")
        print(f"   경로: {exp_dir}/")

    # ========== 일반 학습 (K-Fold 미사용) ==========
    else:
        # 데이터
        train_loader, val_loader = get_dataloaders(cfg)

        # 모델
        print("🔨 모델 생성...")
        model = get_model(cfg.MODEL_NAME, cfg.NUM_CLASSES, cfg.DROPOUT)
        model = model.to(cfg.DEVICE)
        print("✅ 모델 준비 완료\n")

        # 학습
        trainer = Trainer(model, cfg, exp_dir)
        trainer.train(train_loader, val_loader)

        print("\n" + "="*70)
        print("🎉 학습 완료!")
        print("="*70)
        print(f"\n💾 Best Val F1: {trainer.best_val_f1:.4f}")
        print(f"   경로: {exp_dir}/best_model.pth")
        print(f"\n다음: python inference.py --exp_dir {exp_dir}\n")

    return exp_dir


if __name__ == '__main__':
    exp_dir = main()