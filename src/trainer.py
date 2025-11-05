"""
src/trainer.py
Label Smoothing 추가
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
import json


class Trainer:
    def __init__(self, model, cfg, exp_dir, fold=None):
        self.model = model
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.fold = fold  # K-Fold 사용 시 Fold 번호
        self.device = cfg.DEVICE
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        # ✅ Label Smoothing 추가!
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1  # 레이블 스무딩
        )
        
        # 로그 파일
        if fold is not None:
            log_filename = f'training_log_fold{fold+1}.txt'
        else:
            log_filename = 'training_log.txt'
        self.log_file = open(exp_dir / log_filename, 'w')
        
        # Best 추적
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.patience = 10  # 7 → 10 (더 기다림)
        
        # History
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_f1': []
        }
    
    def log_print(self, msg):
        """화면과 파일에 동시 출력"""
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
    
    def train_epoch(self, train_loader, epoch_num):
        """1 Epoch 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        fold_str = f'Fold {self.fold+1} ' if self.fold is not None else ''
        pbar = tqdm(train_loader, desc=f'{fold_str}Epoch {epoch_num}/{self.cfg.EPOCHS} [Train]')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, epoch_num):
        """Validation with progress bar"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        fold_str = f'Fold {self.fold+1} ' if self.fold is not None else ''
        pbar = tqdm(val_loader, desc=f'{fold_str}Epoch {epoch_num}/{self.cfg.EPOCHS} [Val]  ')
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.3f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, f1
    
    def train(self, train_loader, val_loader):
        """전체 학습 루프"""
        self.log_print("="*70)
        self.log_print("🔥 학습 시작!")
        self.log_print("="*70)
        self.log_print("")
        
        for epoch in range(self.cfg.EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch + 1)
            
            # Validation
            val_loss, val_f1 = self.validate(val_loader, epoch + 1)
            
            # 기록
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            # 로그 출력
            self.log_print(f"\nEpoch {epoch+1}/{self.cfg.EPOCHS}:")
            self.log_print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            self.log_print(f"  Val Loss:   {val_loss:.4f} | Val F1:    {val_f1:.4f}")
            
            # Best 모델 저장
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0

                if self.fold is not None:
                    model_path = self.exp_dir / f'best_model_fold{self.fold+1}.pth'
                else:
                    model_path = self.exp_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), model_path)
                self.log_print(f"  ✨ Best!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.log_print(f"\n⏹️  Early Stopping at Epoch {epoch+1}")
                    break
        
        self.log_file.close()
        self.save_results()
    
    def save_results(self):
        """결과를 JSON 파일로 저장"""
        results = {
            'timestamp': self.exp_dir.name.replace('exp_', ''),
            'fold': self.fold + 1 if self.fold is not None else None,
            'config': {
                'model_name': self.cfg.MODEL_NAME,
                'image_size': self.cfg.IMAGE_SIZE,
                'batch_size': self.cfg.BATCH_SIZE,
                'epochs': self.cfg.EPOCHS,
                'lr': self.cfg.LR,
                'dropout': self.cfg.DROPOUT,
                'weight_decay': self.cfg.WEIGHT_DECAY
            },
            'best_results': {
                'val_f1': self.best_val_f1
            },
            'history': self.history
        }

        if self.fold is not None:
            results_path = self.exp_dir / f'results_fold{self.fold+1}.json'
        else:
            results_path = self.exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📊 결과 저장: {results_path}")
