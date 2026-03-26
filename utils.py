import os
import numpy as np
import torch
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from config import Config
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, f1_score
import pandas as pd
import time

class MetricsLogger:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(Config.LOG_PATH, f'training_metrics_{self.timestamp}.txt')
        self.csv_path = os.path.join(Config.CSV_PATH, Config.METRICS_FILE.format(timestamp=self.timestamp))
        self.current_fold = 0
        self.fold_start_time = None
        
        # Initialize CSV file with headers
        self.fold_metrics = []
        headers = ['Fold', 'ACC', 'AUC', 'AUPR', 'F1', 'Precision', 'Recall', 'MCC', 
                  'Specificity', 'FPR', 'FDR', 'TN', 'FP', 'FN', 'TP', 'Total', 'Training Time (s)']
        pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        
        with open(self.log_path, 'w') as f:
            f.write(f'Training Metrics Log - Started at {self.timestamp}\n')

    def set_fold(self, fold):
        self.current_fold = fold
        self.fold_start_time = time.time()  # 记录折开始时间
        with open(self.log_path, 'a') as f:
            f.write(f'\nStarting Fold {fold + 1}/{Config.CV_CONFIG["n_splits"]}\n')

    def log_metrics(self, epoch, metrics):
        """Log training and validation metrics for each epoch"""
        metrics_str = (
            f"Fold {self.current_fold + 1} | "
            f"epoch:{epoch+1:04d} | "
            f"train_loss:{metrics.get('train_loss', 0):.4f} | "
            f"train_acc:{metrics.get('train_ACC', 0)*100:.2f}% | "
            f"train_auc:{metrics.get('train_AUC', 0):.4f} | "
            f"train_prec:{metrics.get('train_Precision', 0)*100:.2f}% | "
            f"train_recall:{metrics.get('train_Recall', 0)*100:.2f}% | "
            f"val_loss:{metrics.get('val_loss', 0):.4f} | "
            f"val_acc:{metrics.get('ACC', 0)*100:.2f}% | "
            f"val_auc:{metrics.get('AUC', 0):.4f} | "
            f"val_prec:{metrics.get('Precision', 0)*100:.2f}% | "
            f"val_recall:{metrics.get('Recall', 0)*100:.2f}%"
        )
        
        logging.info(metrics_str)
        with open(self.log_path, 'a') as f:
            f.write(metrics_str + '\n')

    def log_fold_metrics(self, fold, metrics):
        """Log fold metrics to CSV file"""
        # Calculate training time
        training_time = time.time() - self.fold_start_time
        
        # Create a metrics dictionary with the required fields
        fold_metrics = {
            'Fold': fold + 1,
            'ACC': metrics['ACC'],
            'AUC': metrics['AUC'],
            'AUPR': metrics['AUPR'],
            'F1': metrics['F1'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'MCC': metrics['MCC'],
            'Specificity': metrics['Specificity'],
            'FPR': metrics['FPR'],
            'FDR': metrics['FDR'],
            'TN': metrics['TN'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'TP': metrics['TP'],
            'Total': metrics['Total'],
            'Training Time (s)': round(training_time, 2)  # 添加训练时间
        }
        
        # Convert to DataFrame and save
        metrics_df = pd.DataFrame([fold_metrics])
        metrics_df.to_csv(self.csv_path, mode='a', header=False, index=False)
        self.fold_metrics.append(fold_metrics)
        
        if fold == Config.CV_CONFIG['n_splits'] - 1:
            # Calculate and log average metrics
            avg_metrics = pd.DataFrame(self.fold_metrics).mean()
            avg_metrics_dict = {
                'Fold': 'Average',
                'ACC': avg_metrics['ACC'],
                'AUC': avg_metrics['AUC'],
                'AUPR': avg_metrics['AUPR'],
                'F1': avg_metrics['F1'],
                'Precision': avg_metrics['Precision'],
                'Recall': avg_metrics['Recall'],
                'MCC': avg_metrics['MCC'],
                'Specificity': avg_metrics['Specificity'],
                'FPR': avg_metrics['FPR'],
                'FDR': avg_metrics['FDR'],
                'TN': avg_metrics['TN'],
                'FP': avg_metrics['FP'],
                'FN': avg_metrics['FN'],
                'TP': avg_metrics['TP'],
                'Total': avg_metrics['Total'],
                'Training Time (s)': avg_metrics['Training Time (s)']  # 添加平均训练时间
            }
            pd.DataFrame([avg_metrics_dict]).to_csv(self.csv_path, mode='a', header=False, index=False)

def calculate_metrics(outputs, labels):
    predictions = (outputs >= 0.5).float()
    tp = ((predictions == 1) & (labels == 1)).float().sum()
    fp = ((predictions == 1) & (labels == 0)).float().sum()
    fn = ((predictions == 0) & (labels == 1)).float().sum()
    tn = ((predictions == 0) & (labels == 0)).float().sum()
    
    # Calculate all required metrics
    total = len(labels)
    acc = (tp + tn) / total
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    fpr = fp / (tn + fp + 1e-8)  # False Positive Rate
    fdr = fp / (tp + fp + 1e-8)  # False Discovery Rate
    
    try:
        auc = roc_auc_score(labels.cpu().numpy(), outputs.detach().cpu().numpy())
    except:
        auc = 0.5
        
    mcc = matthews_corrcoef(labels.cpu().numpy(), predictions.cpu().numpy())
    
    # Calculate AUPR
    precision_vals, recall_vals, _ = precision_recall_curve(labels.cpu().numpy(), outputs.detach().cpu().numpy())
    aupr = np.trapz(recall_vals, precision_vals)
    
    # Calculate F1
    f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())
    
    metrics = {
        'ACC': acc.item(),
        'AUC': auc,
        'AUPR': aupr,
        'F1': f1,
        'Precision': precision.item(),
        'Recall': recall.item(),
        'MCC': mcc,
        'Specificity': specificity.item(),
        'FPR': fpr.item(),
        'FDR': fdr.item(),
        'TN': tn.item(),
        'FP': fp.item(),
        'FN': fn.item(),
        'TP': tp.item(),
        'Total': total
    }
    
    return metrics

def load_training_data():
    """加载训练数据
    
    根据Config.FEATURE_CONFIG中的配置加载指定的特征数据，并确保所有特征格式一致
    """
    data = {}
    active_features = Config.FEATURE_CONFIG['active_features']
    feature_info = Config.FEATURE_CONFIG['feature_info']
    
    for feature_name in active_features:
        if feature_name not in feature_info:
            raise ValueError(f"Feature '{feature_name}' not found in feature_info configuration")
            
        file_path = os.path.join(Config.DATA_PATH, feature_info[feature_name]['path'])
        feature_data = np.load(file_path)
        
        # 特殊处理structure_score特征
        if feature_name == 'structure_score':
            # 将structure_score从(N, 487)转换为(N, 1, 487)格式
            feature_data = np.expand_dims(feature_data, axis=1)
        
        data[feature_name] = torch.FloatTensor(feature_data)
        
        # 打印每个特征的形状以便调试
        logging.info(f"Feature {feature_name} shape: {data[feature_name].shape}")
        
    labels = torch.FloatTensor(np.load(f'{Config.DATA_PATH}/train_label.npy'))
    
    logging.info(f"Loaded data - Total samples: {len(labels)}")
    logging.info(f"Active features: {', '.join(active_features)}")
    logging.info(f"Class distribution - Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)}")
    
    return data, labels

def create_data_loaders(data, labels, fold_indices=None):
    """创建训练和验证数据加载器"""
    if fold_indices is None:
        # Original train-val split logic
        dataset = TensorDataset(*data.values(), labels)
        train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
    else:
        # Cross-validation split
        train_idx, val_idx = fold_indices
        train_data = {k: v[train_idx] for k, v in data.items()}
        val_data = {k: v[val_idx] for k, v in data.items()}
        
        train_dataset = TensorDataset(*train_data.values(), labels[train_idx])
        val_dataset = TensorDataset(*val_data.values(), labels[val_idx])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        num_workers=Config.NUM_WORKERS
    )
    
    return train_loader, val_loader 

def save_predictions(predictions, labels):
    """保存预测分数到CSV文件
    
    Args:
        predictions: 模型预测分数
        labels: 真实标签
    """
    
    # 创建DataFrame
    df = pd.DataFrame({
        'true_label': labels.numpy(),
        f'prediction': predictions  # 使用特征名称
    })
    
    # 生成文件名
    filename = os.path.join(
        Config.RESULT_PATH, 
        f'predictions.csv'  # 使用特征名称
    )
    
    # 保存CSV文件
    df.to_csv(filename, index=True)
    logging.info(f"Predictions saved to {filename}") 