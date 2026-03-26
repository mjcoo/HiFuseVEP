import os
import torch
import logging
from tqdm import tqdm
from model import get_model, DynamicBalancedFocalLoss
from config import Config
from utils import MetricsLogger, calculate_metrics, load_training_data, create_data_loaders, save_predictions
from sklearn.model_selection import KFold
import numpy as np
import time

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def train_fold(fold, train_loader, val_loader, logger):
    device = Config.DEVICE
    
    # Initialize model and training components
    logging.info(f"Initializing model and training components for fold {fold+1}...")
    #print("####")
    model, optimizer, scheduler = get_model()
    model = model.to(device)

    criterion = DynamicBalancedFocalLoss(
        gamma=Config.FOCAL_LOSS['gamma'],
        alpha=Config.FOCAL_LOSS['alpha'],
        l1_lambda=Config.LOSS['l1_lambda'],
        label_smoothing=Config.LOSS['label_smoothing']
    ).to(device)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_auc = float('-inf')  # 改为负无穷，因为我们要最大化AUC
    patience_counter = 0
    best_val_metrics = None
    best_predictions = None

    for epoch in range(Config.MAX_EPOCHS):
        logging.info(f"\nFold {fold+1} - Epoch {epoch+1}/{Config.MAX_EPOCHS}")
        model.train()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics, val_preds = validate(model, val_loader, criterion, device, epoch)
        
        # Combine metrics for logging with proper prefixes for training metrics
        epoch_metrics = {
            'train_loss': train_metrics['train_loss'],
            'train_ACC': train_metrics['ACC'],
            'train_AUC': train_metrics['AUC'],
            'train_Precision': train_metrics['Precision'],
            'train_Recall': train_metrics['Recall'],
            'val_loss': val_metrics['val_loss'],
            **val_metrics
        }
        
        scheduler.step()
        logger.log_metrics(epoch, epoch_metrics)
        
        
        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            best_val_metrics = val_metrics
            best_predictions = val_preds
            torch.save(model.state_dict(), f'{Config.MODEL_PATH}/best_model_fold_{fold+1}.pt')
            logging.info("Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # 记录训练时间
    logging.info(f"Fold {fold+1} training completed in {time.time() - logger.fold_start_time:.2f} seconds")
    
    return best_val_metrics, best_predictions

def train():
    # Load data
    data, labels = load_training_data()
    
    # Initialize K-fold cross validation
    kf = KFold(**Config.CV_CONFIG)
    logger = MetricsLogger()
    
    logging.info(f"Starting {Config.CV_CONFIG['n_splits']}-fold cross validation...")

    # 创建数组来存储所有样本的预测分数
    all_predictions = np.zeros(len(labels))
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(next(iter(data.values())))):
        logger.set_fold(fold)  # Set current fold
        logging.info(f"\nTraining Fold {fold+1}/{Config.CV_CONFIG['n_splits']}")
        
        # Create data loaders for this fold
        train_loader, val_loader = create_data_loaders(
            data, labels, fold_indices=(train_idx, val_idx)
        )
        
        # Train the fold
        best_metrics, val_predictions = train_fold(fold, train_loader, val_loader, logger)

        # 将该fold的预测结果存储到对应位置
        all_predictions[val_idx] = val_predictions
        
        # Log fold metrics
        logger.log_fold_metrics(fold, best_metrics)
    
    # 保存预测分数到CSV文件
    save_predictions(all_predictions, labels)
    logging.info("Cross-validation completed!")

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        *inputs, labels = [x.to(device) for x in batch]
       
        outputs,con_loss= model(inputs)
        
        loss = criterion(labels, outputs, model)+con_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_outputs.extend(outputs.detach().cpu())
        all_labels.extend(labels.cpu())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = calculate_metrics(torch.tensor(all_outputs), torch.tensor(all_labels))
    metrics['train_loss'] = total_loss / len(loader)
    return metrics

def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating', leave=False)
        for batch in pbar:
            *inputs, labels = [x.to(device) for x in batch]
            outputs, con_loss = model(inputs)
           
            loss = criterion(labels, outputs, model)+con_loss
            total_loss += loss.item()
            all_outputs.extend(outputs.cpu())
            all_labels.extend(labels.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_outputs = torch.tensor(all_outputs)
    all_labels = torch.tensor(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)
    metrics['val_loss'] = total_loss / len(loader)
    return metrics, all_outputs.numpy()

if __name__ == "__main__":
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    os.makedirs(Config.LOG_PATH, exist_ok=True)
    os.makedirs(Config.CSV_PATH, exist_ok=True)
    os.makedirs(Config.RESULT_PATH, exist_ok=True)
    
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
        logging.info("CUDA is available, using GPU")
    else:
        logging.info("CUDA is not available, using CPU")
    
    train() 