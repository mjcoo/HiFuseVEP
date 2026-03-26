    # HiFuseVEP Model

This project implements a deep learning-based protein sequence classification model using various protein features such as ProtT5, ESM, and others for VEP tasks.
<img width="1757" height="908" alt="image" src="https://github.com/user-attachments/assets/fd389ee6-acf7-49ad-9c4b-5d966efcbddf" />

## Project Overview


This project aims to utilize multiple state-of-the-art protein embedding representation methods (including local and global features) to build an VEP model. Main features include:

- Multi-modal feature fusion: Integrates multiple protein features such as ProtT5, ESM, etc.
- Transformer encoder: Custom encoder processing protein sequences
- Dynamic Balanced Focal Loss: Address class imbalance problems
- Cross-validation: Uses K-fold cross-validation to ensure model generalization

## File Structure

```
├── Encoder.py         # Transformer encoder implementation
├── model.py           # Main model architecture definition
├── config.py          # Project configuration and hyperparameters
├── train.py           # Training script
├── test.py            # Testing script
├── utils.py           # Utility functions and helper code
└── README.md          # Project documentation
```

## Tech Stack

- Python 3.x
- PyTorch 2.x
- scikit-learn
- NumPy
- PyTorch Metric Learning

## Model Architecture

The model mainly includes the following components:

1. **Feature Processing Module**: Processes protein features from different sources
2. **Encoder**: Transformer encoder with multi-head attention mechanism and positional encoding
3. **Feature Adaptive Fusion Module**: Dynamic gating mechanism to fuse multi-modal features
4. **Classifier**: Dynamic classifier for final prediction

## Features

- **Multi-feature Adaptive Fusion**: Supports various protein features (ESM1b, ESMC, ProtT5, etc.)
- **Dynamic Balanced Loss**: Combines Focal Loss and label smoothing to address data imbalance
- **Attention Mechanism**: Implements channel attention mechanism to improve model performance
- **Weight Standardization**: Uses weight-standardized convolutions for training stability
- **DropPath Regularization**: Prevents overfitting and improves generalization

## Configuration

All project configurations are in `config.py`, including:
- Model parameters (layers, hidden dimensions, etc.)
- Training parameters (batch size, learning rate, etc.)
- Optimizer parameters
- Feature selection configuration
- Cross-validation parameters

## Training

Run the training script to start training the model:

```bash
python train.py
```

During training:
- Automatically loads specified protein feature data
- Performs K-fold cross-validation
- Records training metrics
- Saves the best model weights
- Early stopping to prevent overfitting

## Usage

1. Make sure to install required dependencies
2. Prepare training data (protein features and labels)
3. Configure relevant parameters in `config.py`
4. Run the training script

## Model Evaluation

Model evaluation uses multiple metrics:
- AUC (Area Under Curve)
- ACC (Accuracy)
- Precision
- Recall

## Notes

- Ensure sufficient GPU memory (if training on GPU)
- Data needs to be preprocessed into the model's expected format
- Adjust hyperparameters according to specific tasks
    
