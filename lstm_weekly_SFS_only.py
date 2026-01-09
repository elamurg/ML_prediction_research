"""
LSTM Model - Weekly Data Version (SFS Only)

This version uses ONLY the Street Fashion Style data without external 
features (no weather, no Google Trends) to reduce noise and complexity.

RATIONALE:
- ~400 samples is small for deep learning
- External features may add noise rather than signal
- Simpler model = less overfitting risk
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    """LSTM Neural Network for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, 
                 num_layers: int = 1, dropout: float = 0.1):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out.squeeze()


def create_sequences(features: np.ndarray, 
                    target: np.ndarray, 
                    sequence_length: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)


class LSTMForecasterSimple:
    """Simplified LSTM for small datasets."""
    
    def __init__(self, sequence_length: int = 8, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.1,
                 learning_rate: float = 0.001, epochs: int = 100,
                 batch_size: int = 16):
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.training_losses = []
        self.feature_names = None
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X_array = X.values
        y_array = y.values.reshape(-1, 1)
        
        if fit_scalers:
            X_scaled = self.feature_scaler.fit_transform(X_array)
            y_scaled = self.target_scaler.fit_transform(y_array).flatten()
        else:
            X_scaled = self.feature_scaler.transform(X_array)
            y_scaled = self.target_scaler.transform(y_array).flatten()
        
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, self.sequence_length)
        return X_seq, y_seq
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              verbose: bool = True) -> 'LSTMForecasterSimple':
        
        self.feature_names = list(X_train.columns)
        X_train_seq, y_train_seq = self.prepare_data(X_train, y_train, fit_scalers=True)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_data(X_val, y_val, fit_scalers=False)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
            y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
        
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        input_size = X_train.shape[1]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.training_losses.append(avg_loss)
            
            val_loss = avg_loss
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor)
                    val_loss = criterion(val_pred, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 25 == 0:
                msg = f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        
        X_scaled = self.feature_scaler.transform(X.values)
        dummy_y = np.zeros(len(X))
        X_seq, _ = create_sequences(X_scaled, dummy_y, self.sequence_length)
        
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        return predictions


def prepare_sfs_only_features(sfs_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Prepare features using ONLY SFS data - no external features.
    
    This creates a minimal feature set:
    - Lag features (past values)
    - Rolling statistics
    - Simple momentum
    - Calendar features
    """
    df = pd.DataFrame(index=sfs_df.index)
    df[target_col] = sfs_df[target_col]
    
    # === LAG FEATURES (fewer lags) ===
    for lag in [1, 2, 4, 8, 12]:
        df[f'lag_{lag}w'] = df[target_col].shift(lag)
    
    # === ROLLING STATISTICS (simpler) ===
    for window in [4, 8, 12]:
        df[f'rolling_mean_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
        df[f'rolling_std_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).std()
    
    # === SIMPLE MOMENTUM ===
    df['momentum_1w'] = df[target_col].diff(1)
    df['momentum_4w'] = df[target_col].diff(4)
    
    # === CALENDAR FEATURES ===
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # === CLEAN UP ===
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


def create_train_test_split(features_df: pd.DataFrame,
                            target_col: str,
                            split_at_peak: bool = True) -> Dict:
    """Create train/test split."""
    if split_at_peak:
        peak_idx = features_df[target_col].idxmax()
        peak_loc = features_df.index.get_loc(peak_idx)
        
        print(f"Peak found at: {peak_idx.strftime('%Y-%m-%d')}")
        print(f"Peak value: {features_df[target_col].max():.2f}")
        
        train_df = features_df.iloc[:peak_loc + 1]
        test_df = features_df.iloc[peak_loc + 1:]
    else:
        split_point = int(len(features_df) * 0.75)
        train_df = features_df.iloc[:split_point]
        test_df = features_df.iloc[split_point:]
    
    feature_cols = [c for c in features_df.columns if c != target_col]
    
    return {
        'X_train': train_df[feature_cols],
        'y_train': train_df[target_col],
        'X_test': test_df[feature_cols],
        'y_test': test_df[target_col],
        'train_dates': train_df.index,
        'test_dates': test_df.index,
    }


def train_and_evaluate_lstm(split_data: Dict, 
                            model_name: str = "LSTM",
                            sequence_length: int = 8,
                            hidden_size: int = 32,
                            epochs: int = 150) -> Dict:
    """Complete LSTM training and evaluation."""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")
    
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")
    print(f"Sequence length: {sequence_length} weeks")
    print(f"Hidden size: {hidden_size}")
    print(f"Device: {device}")
    
    model = LSTMForecasterSimple(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.1,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=16
    )
    
    print("\nTraining LSTM...")
    model.train(X_train, y_train, X_test, y_test, verbose=True)
    
    # Prediction
    bridge_length = sequence_length
    X_bridge = pd.concat([X_train.iloc[-bridge_length:], X_test])
    
    predictions = model.predict(X_bridge)
    y_actual = y_test.values
    
    if len(predictions) > len(y_actual):
        predictions = predictions[-len(y_actual):]
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    mae = mean_absolute_error(y_actual, predictions)
    r2 = r2_score(y_actual, predictions)
    
    mask = y_actual != 0
    mape = np.mean(np.abs((y_actual[mask] - predictions[mask]) / y_actual[mask])) * 100 if mask.sum() > 0 else np.nan
    
    print("\n--- Test Performance ---")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R2:   {r2:.3f}")
    print(f"  MAPE: {mape:.1f}%")
    
    return {
        'model': model,
        'test_metrics': {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape},
        'predictions': predictions,
        'actual': y_actual,
        'test_dates': split_data['test_dates'],
        'train_dates': split_data['train_dates'],
        'y_train': y_train
    }


def plot_predictions(results: Dict, title: str = None, save_path: str = None):
    """Visualize predictions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(results['train_dates'], results['y_train'], color='#2A9D8F', 
            linewidth=1.5, label='Training (Actual)', alpha=0.8)
    ax.plot(results['test_dates'], results['actual'],
            color='#457B9D', linewidth=1.5, label='Test (Actual)')
    ax.plot(results['test_dates'], results['predictions'],
            color='#E63946', linewidth=1.5, linestyle='--',
            label='Test (Predicted)', alpha=0.8)
    ax.axvline(x=results['train_dates'][-1], color='black', 
               linestyle='--', alpha=0.5, label='Train/Test Split')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    metrics = results['test_metrics']
    ax.annotate(f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.2f}", 
                xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import os
    
    os.makedirs('plots', exist_ok=True)
    print(f"Using device: {device}")
    
    # Load SFS data only
    print("\n" + "="*60)
    print("LOADING SFS DATA ONLY")
    print("="*60)
    
    sfs_df = pd.read_csv('data/weekly_trend_counts.csv', parse_dates=['date'], index_col='date')
    sfs_df['zara_frequency'] = sfs_df['zara_dress_interp']
    sfs_df['chanel_frequency'] = sfs_df['chanel_bag_interp']
    
    print(f"Loaded {len(sfs_df)} weeks of data")
    
    # === ZARA DRESS ===
    print("\n" + "="*60)
    print("ZARA DRESS - LSTM (SFS ONLY)")
    print("="*60)
    
    zara_features = prepare_sfs_only_features(sfs_df, 'zara_frequency')
    print(f"Features: {len(zara_features.columns) - 1}")
    
    zara_split = create_train_test_split(zara_features, 'zara_frequency')
    
    zara_results = train_and_evaluate_lstm(
        zara_split,
        model_name="Zara LSTM (SFS Only)",
        sequence_length=8,
        hidden_size=32,
        epochs=150
    )
    
    plot_predictions(zara_results,
        title='Zara Dress: LSTM (SFS Only) Predictions',
        save_path='plots/zara_lstm_sfs_only.png'
    )
    
    # === CHANEL BAG ===
    print("\n" + "="*60)
    print("CHANEL BAG - LSTM (SFS ONLY)")
    print("="*60)
    
    chanel_features = prepare_sfs_only_features(sfs_df, 'chanel_frequency')
    chanel_split = create_train_test_split(chanel_features, 'chanel_frequency')
    
    chanel_results = train_and_evaluate_lstm(
        chanel_split,
        model_name="Chanel LSTM (SFS Only)",
        sequence_length=8,
        hidden_size=32,
        epochs=150
    )
    
    plot_predictions(chanel_results,
        title='Chanel Bag: LSTM (SFS Only) Predictions',
        save_path='plots/chanel_lstm_sfs_only.png'
    )
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("LSTM (SFS ONLY) SUMMARY")
    print("="*60)
    print(f"\nZara:   R² = {zara_results['test_metrics']['r2']:.3f}, RMSE = {zara_results['test_metrics']['rmse']:.2f}")
    print(f"Chanel: R² = {chanel_results['test_metrics']['r2']:.3f}, RMSE = {chanel_results['test_metrics']['rmse']:.2f}")