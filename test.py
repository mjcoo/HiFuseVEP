import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import get_model
from config import Config
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_curve, f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_test_data():
    """Load test data with the same format as training data"""
    data = {}
    active_features = Config.TEST_FEATURE_CONFIG['active_features']
    feature_info = Config.TEST_FEATURE_CONFIG['feature_info']
    
    for feature_name in active_features:
        if feature_name not in feature_info:
            raise ValueError(f"Feature '{feature_name}' not found in feature_info configuration")
            
        file_path = os.path.join(Config.TEST_DATA_PATH, feature_info[feature_name]['path'])
        feature_data = np.load(file_path)
        
        # Special handling for structure_score feature
        if feature_name == 'structure_score':
            feature_data = np.expand_dims(feature_data, axis=1)
        
        data[feature_name] = torch.FloatTensor(feature_data)
        logging.info(f"Loaded test feature {feature_name} with shape: {data[feature_name].shape}")
        
    labels = torch.FloatTensor(np.load(f'{Config.TEST_DATA_PATH}/test_label.npy'))
    
    logging.info(f"Loaded test data - Total samples: {len(labels)}")
    logging.info(f"Class distribution - Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)}")
    
    return data, labels


def create_test_loader(test_data, test_labels):
    """Create test data loader"""
    test_dataset = TensorDataset(*test_data.values(), test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    return test_loader


def calculate_metrics(outputs, labels):
    """Calculate evaluation metrics
    
    Args:
        outputs: Model prediction probabilities (numpy array or torch tensor)
        labels: Ground truth labels (numpy array or torch tensor)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Convert to numpy if tensor
    if isinstance(outputs, torch.Tensor):
        outputs_np = outputs.detach().cpu().numpy()
    else:
        outputs_np = outputs
        
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels
    
    # Binary predictions with threshold 0.5
    predictions = (outputs_np >= 0.5).astype(float)
    
    # Calculate confusion matrix elements
    tp = ((predictions == 1) & (labels_np == 1)).sum()
    fp = ((predictions == 1) & (labels_np == 0)).sum()
    fn = ((predictions == 0) & (labels_np == 1)).sum()
    tn = ((predictions == 0) & (labels_np == 0)).sum()
    
    total = len(labels_np)
    acc = (tp + tn) / total
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    fpr = fp / (tn + fp + 1e-8)
    fdr = fp / (tp + fp + 1e-8)
    
    # AUC
    try:
        auc = roc_auc_score(labels_np, outputs_np)
    except:
        auc = 0.5
    
    # MCC
    mcc = matthews_corrcoef(labels_np, predictions)
    
    # AUPR
    precision_vals, recall_vals, _ = precision_recall_curve(labels_np, outputs_np)
    aupr = np.trapz(recall_vals, precision_vals)
    
    # F1
    f1 = f1_score(labels_np, predictions)
    
    metrics = {
        'ACC': float(acc),
        'AUC': float(auc),
        'AUPR': float(aupr),
        'F1': float(f1),
        'Precision': float(precision),
        'Recall': float(recall),
        'MCC': float(mcc),
        'Specificity': float(specificity),
        'FPR': float(fpr),
        'FDR': float(fdr),
        'TN': float(tn),
        'FP': float(fp),
        'FN': float(fn),
        'TP': float(tp),
        'Total': float(total)
    }
    
    return metrics


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        device: Computing device (CPU/GPU)
    
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Numpy array of prediction probabilities
    """
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            *inputs, labels = [x.to(device) for x in batch]
            outputs, _ = model(inputs)  # Ignore contrastive loss for testing
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)
    
    return metrics, all_outputs, all_labels


def save_fold_results(predictions, labels, metrics, fold):
    """Save individual fold test results
    
    Args:
        predictions: Prediction probabilities
        labels: Ground truth labels
        metrics: Dictionary of metrics
        fold: Fold number
    """
    # Save predictions
    predictions_path = os.path.join(Config.TEST_RESULT_PATH, f'test_predictions_fold_{fold}.csv')
    predictions_df = pd.DataFrame({
        'true_label': labels,
        'prediction_prob': predictions,
        'prediction_class': (predictions >= 0.5).astype(int)
    })
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join(Config.TEST_RESULT_PATH, f'test_metrics_fold_{fold}.csv')
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    
    logging.info(f"Fold {fold} results saved to {predictions_path}")


def save_aggregated_results(all_fold_metrics, all_fold_predictions, labels):
    """Save aggregated results from all folds
    
    Args:
        all_fold_metrics: List of metric dictionaries from each fold
        all_fold_predictions: List of prediction arrays from each fold
        labels: Ground truth labels
    """
    # ============================================================
    # 1. Calculate mean and std for each metric across all folds
    # ============================================================
    metrics_df = pd.DataFrame(all_fold_metrics)
    
    # Define the 8 primary evaluation metrics (as mentioned in your description)
    primary_metrics = ['ACC', 'AUC', 'AUPR', 'F1', 'Precision', 'Recall', 'MCC', 'Specificity']
    
    summary_stats = {}
    for metric in metrics_df.columns:
        values = metrics_df[metric].values
        summary_stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # ============================================================
    # 2. Save detailed metrics for all folds
    # ============================================================
    all_folds_metrics_path = os.path.join(Config.TEST_RESULT_PATH, 'all_folds_metrics.csv')
    metrics_df.insert(0, 'Fold', range(1, len(all_fold_metrics) + 1))
    metrics_df.to_csv(all_folds_metrics_path, index=False)
    logging.info(f"All folds metrics saved to {all_folds_metrics_path}")
    
    # ============================================================
    # 3. Save summary statistics (Mean ± Std)
    # ============================================================
    summary_path = os.path.join(Config.TEST_RESULT_PATH, 'metrics_summary.csv')
    summary_rows = []
    for metric in primary_metrics:
        if metric in summary_stats:
            stats = summary_stats[metric]
            summary_rows.append({
                'Metric': metric,
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Mean±Std': f"{stats['mean']:.4f}±{stats['std']:.4f}",
                'Min': stats['min'],
                'Max': stats['max']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Metrics summary saved to {summary_path}")
    
    # ============================================================
    # 4. Calculate ensemble predictions (average of all folds)
    # ============================================================
    ensemble_predictions = np.mean(all_fold_predictions, axis=0)
    ensemble_metrics = calculate_metrics(ensemble_predictions, labels)
    
    # Save ensemble predictions
    ensemble_pred_path = os.path.join(Config.TEST_RESULT_PATH, 'ensemble_predictions.csv')
    ensemble_pred_df = pd.DataFrame({
        'true_label': labels,
        'ensemble_prediction_prob': ensemble_predictions,
        'ensemble_prediction_class': (ensemble_predictions >= 0.5).astype(int)
    })
    # Add individual fold predictions for reference
    for i, fold_pred in enumerate(all_fold_predictions):
        ensemble_pred_df[f'fold_{i+1}_prob'] = fold_pred
    ensemble_pred_df.to_csv(ensemble_pred_path, index=False)
    logging.info(f"Ensemble predictions saved to {ensemble_pred_path}")
    
    # Save ensemble metrics
    ensemble_metrics_path = os.path.join(Config.TEST_RESULT_PATH, 'ensemble_metrics.csv')
    ensemble_metrics_df = pd.DataFrame([ensemble_metrics])
    ensemble_metrics_df.to_csv(ensemble_metrics_path, index=False)
    logging.info(f"Ensemble metrics saved to {ensemble_metrics_path}")
    
    return summary_stats, ensemble_metrics


def print_final_results(summary_stats, ensemble_metrics, n_folds):
    """Print final evaluation results in a formatted manner
    
    Args:
        summary_stats: Dictionary containing mean/std for each metric
        ensemble_metrics: Metrics calculated from ensemble predictions
        n_folds: Number of folds used
    """
    primary_metrics = ['ACC', 'AUC', 'AUPR', 'F1', 'Precision', 'Recall', 'MCC', 'Specificity']
    
    logging.info("\n" + "=" * 70)
    logging.info(f"FINAL EVALUATION RESULTS ({n_folds}-Fold Cross-Validation)")
    logging.info("=" * 70)
    
    # Print mean ± std for each metric
    logging.info("\n--- Average Performance Across All Folds (Mean ± Std) ---")
    logging.info("-" * 50)
    for metric in primary_metrics:
        if metric in summary_stats:
            mean = summary_stats[metric]['mean']
            std = summary_stats[metric]['std']
            logging.info(f"{metric:15s}: {mean:.4f} ± {std:.4f}")
    logging.info("-" * 50)
    
    # Print ensemble performance
    logging.info("\n--- Ensemble Model Performance (Averaged Predictions) ---")
    logging.info("-" * 50)
    for metric in primary_metrics:
        if metric in ensemble_metrics:
            logging.info(f"{metric:15s}: {ensemble_metrics[metric]:.4f}")
    logging.info("-" * 50)
    
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION COMPLETE")
    logging.info("=" * 70)


def test():
    """Main testing function implementing 10-fold cross-validation evaluation strategy
    
    This function:
    1. Loads the independent test set
    2. Evaluates each of the 10 fold models on the test set
    3. Calculates mean ± std of metrics across all folds
    4. Generates ensemble predictions by averaging predictions from all folds
    """
    # Setup device
    device = Config.DEVICE
    logging.info(f"Using device: {device}")
    
    # Load test data
    logging.info("Loading test data...")
    test_data, test_labels = load_test_data()
    test_loader = create_test_loader(test_data, test_labels)
    
    # Convert labels to numpy for later use
    test_labels_np = test_labels.numpy()
    
    # Initialize model architecture
    logging.info("Initializing model architecture...")
    model, _, _ = get_model()
    model = model.to(device)
    
    # Storage for all folds' results
    all_fold_metrics = []
    all_fold_predictions = []
    valid_folds = []
    
    n_splits = Config.CV_CONFIG['n_splits']
    
    # ============================================================
    # Evaluate each fold's model on the independent test set
    # ============================================================
    logging.info(f"\nEvaluating {n_splits} fold models on independent test set...")
    logging.info("=" * 60)
    
    for fold in range(1, n_splits + 1):
        model_path = f'{Config.MODEL_PATH}/best_model_fold_{fold}.pt'
        
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found: {model_path}, skipping fold {fold}")
            continue
        
        # Load model weights
        logging.info(f"\n[Fold {fold}/{n_splits}] Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate on test set
        metrics, predictions, _ = evaluate_model(model, test_loader, device)
        
        # Store results
        all_fold_metrics.append(metrics)
        all_fold_predictions.append(predictions)
        valid_folds.append(fold)
        
        # Save individual fold results
        save_fold_results(predictions, test_labels_np, metrics, fold)
        
        # Print fold metrics
        logging.info(f"Fold {fold} Test Results:")
        logging.info(f"  ACC: {metrics['ACC']:.4f} | AUC: {metrics['AUC']:.4f} | "
                    f"AUPR: {metrics['AUPR']:.4f} | F1: {metrics['F1']:.4f} | "
                    f"MCC: {metrics['MCC']:.4f}")
    
    # ============================================================
    # Validate that we have results from all expected folds
    # ============================================================
    if len(valid_folds) == 0:
        raise ValueError("No valid models found for evaluation!")
    
    if len(valid_folds) < n_splits:
        logging.warning(f"Only {len(valid_folds)}/{n_splits} fold models were evaluated!")
    
    logging.info(f"\nSuccessfully evaluated {len(valid_folds)} fold models")
    
    # ============================================================
    # Aggregate results and calculate final metrics
    # ============================================================
    logging.info("\nAggregating results from all folds...")
    summary_stats, ensemble_metrics = save_aggregated_results(
        all_fold_metrics, 
        all_fold_predictions, 
        test_labels_np
    )
    
    # ============================================================
    # Print final formatted results
    # ============================================================
    print_final_results(summary_stats, ensemble_metrics, len(valid_folds))
    
    return summary_stats, ensemble_metrics


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(Config.TEST_RESULT_PATH, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    # Run testing
    summary_stats, ensemble_metrics = test()
