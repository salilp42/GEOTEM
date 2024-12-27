"""
Main training script for ECG classification models.
Author: Salil Patel
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score,
                           recall_score, f1_score, confusion_matrix,
                           matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
import logging
import json
from tqdm import tqdm
from scipy.stats import wilcoxon

from geotem.models.architectures import (
    GTN_CrossAttention, 
    BaselineCNN, 
    BaselineLSTM
)
from geotem.features.geometric_features import AdvancedGeometricFeatures
from geotem.visualization.interpretability import (
    plot_gtn_time_attention,
    plot_cnn_gradcam,
    plot_lstm_integrated_gradients
)

# Plot style
plt.rcParams.update({
    'figure.dpi': 300,
    'axes.grid': False,
    'font.size': 12
})
sns.set_style("white")

###############################################################################
# Dataset Class
###############################################################################
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

###############################################################################
# Training & Evaluation Functions
###############################################################################
def train_one_fold(model, train_loader, val_loader,
                   criterion, optimizer, device, epochs, logger, model_name):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        desc_ = f"{model_name} [Epoch {epoch+1}/{epochs}]"
        pbar = tqdm(total=len(train_loader), desc=desc_)
        
        model.train()
        total_train_loss = 0
        for (Xb, yb) in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.update(1)
        
        pbar.close()
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_ = 0
        with torch.no_grad():
            for (Xv, yv) in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                outv = model(Xv).squeeze()
                l_ = criterion(outv, yv)
                val_loss_ += l_.item()
        val_loss_ /= len(val_loader)
        
        logger.info(
            f"{model_name} Epoch {epoch+1}/{epochs} => "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_:.4f}"
        )
        
        if val_loss_ < best_val_loss:
            best_val_loss = val_loss_

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (Xb, yb) in test_loader:
            Xb = Xb.to(device)
            out_ = model(Xb).squeeze().cpu().numpy()
            all_preds.extend(out_)
            all_labels.extend(yb.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    bin_preds = (all_preds >= 0.5).astype(int)
    
    metrics_ = {
        'accuracy': accuracy_score(all_labels, bin_preds),
        'precision': precision_score(all_labels, bin_preds, zero_division=0),
        'recall': recall_score(all_labels, bin_preds, zero_division=0),
        'f1': f1_score(all_labels, bin_preds, zero_division=0),
        'auc': roc_auc,
        'mcc': matthews_corrcoef(all_labels, bin_preds)
    }
    return metrics_, fpr, tpr, all_preds, all_labels

###############################################################################
# Main
###############################################################################
def main():
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading ECG200 dataset...")
    data_dir = Path("data/ECG200")
    train_path = data_dir / "ECG200_TRAIN.txt"
    test_path = data_dir / "ECG200_TEST.txt"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "ECG200_TRAIN.txt or ECG200_TEST.txt not found in data/ECG200/"
        )
    
    train_data = pd.read_csv(train_path, header=None, sep=r"\s+").values
    test_data = pd.read_csv(test_path, header=None, sep=r"\s+").values
    
    X_train_full = train_data[:, 1:]
    y_train_full = (train_data[:, 0] == 1).astype(int)
    X_test = test_data[:, 1:]
    y_test = (test_data[:, 0] == 1).astype(int)

    # Cross-validation
    logger.info("Starting 5-fold cross-validation...")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    epochs_cv = 30
    batch_size = 32
    lr = 1e-3

    results_dict = {
        'GTN': {'metrics': [], 'fprs': [], 'tprs': [], 'aucs': []},
        'CNN': {'metrics': [], 'fprs': [], 'tprs': [], 'aucs': []},
        'LSTM':{'metrics': [], 'fprs': [], 'tprs': [], 'aucs': []}
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
        logger.info(f"\n========== CV FOLD {fold}/{n_splits} ==========")
        X_tr_raw = X_train_full[train_idx]
        y_tr = y_train_full[train_idx]
        X_val_raw = X_train_full[val_idx]
        y_val = y_train_full[val_idx]

        # Extract geometric features
        gf = AdvancedGeometricFeatures()
        X_tr_geo = gf.extract_features(X_tr_raw, fit=True)
        X_val_geo = gf.extract_features(X_val_raw, fit=False)

        # Create datasets
        train_ds_geo = ECGDataset(X_tr_geo, y_tr)
        val_ds_geo = ECGDataset(X_val_geo, y_val)
        train_ds_raw = ECGDataset(X_tr_raw, y_tr)
        val_ds_raw   = ECGDataset(X_val_raw, y_val)

        train_loader_geo = DataLoader(train_ds_geo, batch_size=batch_size, shuffle=True)
        val_loader_geo   = DataLoader(val_ds_geo, batch_size=batch_size)
        train_loader_raw = DataLoader(train_ds_raw, batch_size=batch_size, shuffle=True)
        val_loader_raw   = DataLoader(val_ds_raw, batch_size=batch_size)

        # Models
        seq_len = X_train_full.shape[1]
        models_ = {
            'GTN': GTN_CrossAttention(N=seq_len, out_channels=32, d_model=64, nhead=4, dropout=0.2),
            'CNN': BaselineCNN(sequence_length=seq_len),
            'LSTM':BaselineLSTM(sequence_length=seq_len)
        }

        for model_name in ['GTN','CNN','LSTM']:
            model = models_[model_name].to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            if model_name == 'GTN':
                train_loader = train_loader_geo
                val_loader   = val_loader_geo
            else:
                train_loader = train_loader_raw
                val_loader   = val_loader_raw

            logger.info(f"Training {model_name} (fold={fold})...")
            train_one_fold(model, train_loader, val_loader,
                           criterion, optimizer, device,
                           epochs_cv, logger, model_name)
            
            metrics_val, fpr_val, tpr_val, _, _ = evaluate_model(model, val_loader, device)
            results_dict[model_name]['metrics'].append(metrics_val)
            results_dict[model_name]['fprs'].append(fpr_val)
            results_dict[model_name]['tprs'].append(tpr_val)
            results_dict[model_name]['aucs'].append(metrics_val['auc'])
            
            logger.info(f"{model_name} Fold {fold} (val set) metrics:")
            for k,v in metrics_val.items():
                logger.info(f"  {k}: {v:.4f}")

    # Save final CV metrics
    final_cv_summary = {}
    for model_name, d_ in results_dict.items():
        df_ = pd.DataFrame(d_['metrics'])
        mean_ = df_.mean()
        std_  = df_.std()
        final_cv_summary[model_name] = {
            'mean': mean_.to_dict(),
            'std':  std_.to_dict()
        }
    
    with open(output_dir / "cv_metrics.json","w") as f:
        json.dump(final_cv_summary, f, indent=4)

    logger.info("\n=== Final Average Metrics (across 5 folds) ===")
    for model_name, st_ in final_cv_summary.items():
        logger.info(f"\nModel: {model_name}")
        for mt in ['accuracy','precision','recall','f1','auc','mcc']:
            mv, sv = st_['mean'][mt], st_['std'][mt]
            logger.info(f"  {mt}: {mv:.4f} ± {sv:.4f}")

    # Train final models on full training set
    logger.info("\nTraining final models on full training set...")
    
    gf_full = AdvancedGeometricFeatures()
    X_train_geo_full = gf_full.extract_features(X_train_full, fit=True)
    X_test_geo_full  = gf_full.extract_features(X_test, fit=False)

    train_ds_geo_full = ECGDataset(X_train_geo_full, y_train_full)
    test_ds_geo_full  = ECGDataset(X_test_geo_full,  y_test)

    train_ds_raw_full = ECGDataset(X_train_full, y_train_full)
    test_ds_raw_full  = ECGDataset(X_test,       y_test)

    train_loader_geo_full = DataLoader(train_ds_geo_full, batch_size=batch_size, shuffle=True)
    test_loader_geo_full  = DataLoader(test_ds_geo_full, batch_size=batch_size)

    train_loader_raw_full = DataLoader(train_ds_raw_full, batch_size=batch_size, shuffle=True)
    test_loader_raw_full  = DataLoader(test_ds_raw_full, batch_size=batch_size)

    epochs_final = 30
    final_models = {}
    final_test_loaders = {}
    final_test_results = {}

    for model_name in ['GTN','CNN','LSTM']:
        logger.info(f"\n=== Final Training => {model_name} ===")
        if model_name == 'GTN':
            model = GTN_CrossAttention(N=seq_len, out_channels=32, d_model=64, nhead=4, dropout=0.2).to(device)
            train_loader = train_loader_geo_full
            test_loader  = test_loader_geo_full
        elif model_name == 'CNN':
            model = BaselineCNN(sequence_length=seq_len).to(device)
            train_loader = train_loader_raw_full
            test_loader  = test_loader_raw_full
        else:
            model = BaselineLSTM(sequence_length=seq_len).to(device)
            train_loader = train_loader_raw_full
            test_loader  = test_loader_raw_full

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs_final):
            desc_ = f"{model_name} [Full-Train Epoch {epoch+1}/{epochs_final}]"
            pbar = tqdm(total=len(train_loader), desc=desc_)
            model.train()
            for (Xb, yb) in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(Xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                pbar.update(1)
            pbar.close()

        metrics_test, fpr_test, tpr_test, test_preds, test_labels = evaluate_model(model, test_loader, device)
        final_models[model_name] = model
        final_test_loaders[model_name] = test_loader
        final_test_results[model_name] = metrics_test

        logger.info(f"Test Set Metrics for {model_name}:")
        for k,v in metrics_test.items():
            logger.info(f"  {k}: {v:.4f}")

    with open(output_dir / "final_test_metrics.json","w") as f:
        json.dump(final_test_results, f, indent=4)

    # Generate interpretability plots
    logger.info("\nGenerating interpretability visualizations...")
    
    if 'GTN' in final_models:
        plot_gtn_time_attention(
            final_models['GTN'], 
            X_test_geo_full, 
            y_test, 
            output_dir, 
            n_samples=3, 
            seq_len=seq_len
        )

    if 'CNN' in final_models:
        plot_cnn_gradcam(
            final_models['CNN'], 
            X_test, 
            y_test, 
            output_dir, 
            n_samples=3
        )

    if 'LSTM' in final_models:
        plot_lstm_integrated_gradients(
            final_models['LSTM'], 
            X_test, 
            y_test, 
            output_dir, 
            n_samples=3
        )

    logger.info(f"\nAll done! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
