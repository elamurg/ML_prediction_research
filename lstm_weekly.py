"""
LSTM Model - Weekly Data Version

This module implements LSTM for fashion trend prediction using WEEKLY data.

KEY FEATURES:
- ~400 samples instead of ~94 (4x more data)
- Includes Google Trends weekly data
- Weather data included
- Feature engineering adapted for weekly patterns
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
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
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
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze()


def create_sequences(features: np.ndarray, 
                    target: np.ndarray, 
                    sequence_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)


class LSTMForecaster:
    """LSTM model wrapper for fashion trend forecasting."""
    
    def __init__(self, sequence_length: int = 12, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
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
        """Prepare data for LSTM: scale and create sequences."""
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
              verbose: bool = True) -> 'LSTMForecaster':
        """Train the LSTM model."""
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
        
        self.training_losses = []
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.training_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                msg = f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}"
                if X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(X_val_tensor)
                        val_loss = criterion(val_pred, y_val_tensor).item()
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
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


def prepare_weekly_features(sfs_df: pd.DataFrame, 
                           weather_df: pd.DataFrame,
                           google_df: pd.DataFrame,
                           target_col: str) -> pd.DataFrame:
    """
    Prepare features for weekly LSTM/TFT models.
    
    Parameters:
    -----------
    sfs_df : DataFrame
        Weekly SFS data with trend frequencies
    weather_df : DataFrame
        Weekly weather data
    google_df : DataFrame
        Weekly Google Trends data
    target_col : str
        Target column name ('zara_frequency' or 'chanel_frequency')
    
    Returns:
    --------
    DataFrame with features and target aligned
    """
    df = pd.DataFrame(index=sfs_df.index)
    df[target_col] = sfs_df[target_col]
    
    # === LAG FEATURES ===
    lag_weeks = [1, 2, 4, 8, 12, 26, 52]
    for lag in lag_weeks:
        df[f'lag_{lag}w'] = df[target_col].shift(lag)
    
    # === ROLLING STATISTICS ===
    windows = [4, 12, 26]
    for window in windows:
        df[f'rolling_mean_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
        df[f'rolling_std_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).std()
        df[f'rolling_min_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).min()
        df[f'rolling_max_{window}w'] = df[target_col].shift(1).rolling(window, min_periods=1).max()
    
    # === MOMENTUM FEATURES ===
    df['momentum_1w'] = df[target_col].diff(1)
    df['momentum_4w'] = df[target_col].diff(4)
    df['momentum_12w'] = df[target_col].diff(12)
    
    for period in [1, 4, 12]:
        pct_change = df[target_col].pct_change(period)
        pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
        df[f'pct_change_{period}w'] = pct_change
    
    # === GOOGLE TRENDS FEATURES ===
    if google_df is not None:
        google_aligned = google_df.reindex(df.index, method='nearest').ffill().bfill()
        
        # Determine which trend column to use based on target
        if 'zara' in target_col.lower():
            trend_col = 'zara_search_interest'
        else:
            trend_col = 'chanel_search_interest'
        
        if trend_col in google_aligned.columns:
            df['search_interest'] = google_aligned[trend_col]
            df['search_interest_lag_1w'] = df['search_interest'].shift(1)
            df['search_interest_lag_4w'] = df['search_interest'].shift(4)
            df['search_interest_rolling_4w'] = df['search_interest'].rolling(4, min_periods=1).mean()
            df['search_interest_momentum'] = df['search_interest'].diff(1)
    
    # === WEATHER FEATURES ===
    if weather_df is not None:
        weather_aligned = weather_df.reindex(df.index, method='nearest').ffill().bfill()
        
        for col in weather_aligned.columns:
            df[col] = weather_aligned[col]
            df[f'{col}_lag_1w'] = weather_aligned[col].shift(1)
            df[f'{col}_lag_4w'] = weather_aligned[col].shift(4)
    
    # === CALENDAR FEATURES ===
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # === CLEAN UP ===
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


def create_weekly_train_test_split(features_df: pd.DataFrame,
                                   target_col: str,
                                   split_at_peak: bool = True) -> Dict:
    """Create train/test split for weekly data."""
    peak_idx = features_df[target_col].idxmax()
    peak_loc = features_df.index.get_loc(peak_idx)
    
    print(f"Peak found at: {peak_idx.strftime('%Y-%m-%d')}")
    print(f"Peak value: {features_df[target_col].max():.2f}")
    
    train_df = features_df.iloc[:peak_loc + 1]
    test_df = features_df.iloc[peak_loc + 1:]
    
    feature_cols = [c for c in features_df.columns if c != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_dates': train_df.index,
        'test_dates': test_df.index,
        'peak_date': peak_idx
    }


def train_and_evaluate_lstm_weekly(split_data: Dict, 
                                   model_name: str = "LSTM",
                                   sequence_length: int = 12,
                                   hidden_size: int = 64,
                                   num_layers: int = 2,
                                   dropout: float = 0.2,
                                   epochs: int = 100) -> Dict:
    """Complete LSTM training and evaluation for weekly data."""
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
    print(f"Device: {device}")
    
    model = LSTMForecaster(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=32
    )
    
    print("\nTraining LSTM...")
    model.train(X_train, y_train, X_test, y_test, verbose=True)
    
    # Bridge for continuous predictions
    bridge_length = sequence_length
    X_bridge = pd.concat([X_train.iloc[-bridge_length:], X_test])
    
    predictions = model.predict(X_bridge)
    y_actual = y_test.values
    
    if len(predictions) > len(y_actual):
        predictions = predictions[-len(y_actual):]
    
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
        'train_dates': split_data['train_dates']
    }


def plot_lstm_predictions_weekly(results: Dict, 
                                 train_y: pd.Series,
                                 train_dates: pd.DatetimeIndex,
                                 title: str = None,
                                 save_path: str = None):
    """Visualize LSTM predictions vs actual for weekly data."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(train_dates, train_y, color='#2A9D8F', 
            linewidth=1.5, label='Training (Actual)', alpha=0.8)
    ax.plot(results['test_dates'], results['actual'],
            color='#457B9D', linewidth=1.5, label='Test (Actual)')
    ax.plot(results['test_dates'], results['predictions'],
            color='#E63946', linewidth=1.5, linestyle='--',
            label='Test (Predicted)', alpha=0.8)
    ax.axvline(x=train_dates[-1], color='black', 
               linestyle='--', alpha=0.5, label='Train/Test Split')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    metrics = results['test_metrics']
    metrics_text = f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.2f}"
    ax.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
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
    from load_data_weekly import load_all_weekly_data
    
    os.makedirs('plots', exist_ok=True)
    print(f"Using device: {device}")
    
    # Load weekly data
    print("\n" + "="*60)
    print("LOADING WEEKLY DATA")
    print("="*60)
    
    data = load_all_weekly_data(
        sfs_metadata_path='data/SFS_metadata.csv',
        weather_path='data/California_weather.csv',
        google_trends_path='data/google_trends_weekly_smoothed.csv'
    )
    
    # === ZARA DRESS ===
    print("\n" + "="*60)
    print("ZARA DRESS - WEEKLY LSTM")
    print("="*60)
    
    zara_features = prepare_weekly_features(
        data['sfs'], data['weather'], data['google'], 'zara_frequency'
    )
    print(f"\nZara features shape: {zara_features.shape}")
    print(f"Feature columns: {len(zara_features.columns) - 1}")
    
    zara_split = create_weekly_train_test_split(
        zara_features, 'zara_frequency', split_at_peak=True
    )
    
    zara_results = train_and_evaluate_lstm_weekly(
        zara_split,
        model_name="Zara LSTM (Weekly SFS + Weather + Google)",
        sequence_length=12,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        epochs=150
    )
    
    plot_lstm_predictions_weekly(
        zara_results,
        zara_split['y_train'],
        zara_split['train_dates'],
        title='Zara Dress: Weekly LSTM Predictions vs Actual',
        save_path='plots/zara_lstm_weekly_predictions.png'
    )
    
    # === CHANEL BAG ===
    print("\n" + "="*60)
    print("CHANEL BAG - WEEKLY LSTM")
    print("="*60)
    
    chanel_features = prepare_weekly_features(
        data['sfs'], data['weather'], data['google'], 'chanel_frequency'
    )
    print(f"\nChanel features shape: {chanel_features.shape}")
    
    chanel_split = create_weekly_train_test_split(
        chanel_features, 'chanel_frequency', split_at_peak=True
    )
    
    chanel_results = train_and_evaluate_lstm_weekly(
        chanel_split,
        model_name="Chanel LSTM (Weekly SFS + Weather + Google)",
        sequence_length=12,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        epochs=150
    )
    
    plot_lstm_predictions_weekly(
        chanel_results,
        chanel_split['y_train'],
        chanel_split['train_dates'],
        title='Chanel Bag: Weekly LSTM Predictions vs Actual',
        save_path='plots/chanel_lstm_weekly_predictions.png'
    )
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("WEEKLY LSTM SUMMARY")
    print("="*60)
    print(f"\nZara:   R² = {zara_results['test_metrics']['r2']:.3f}, RMSE = {zara_results['test_metrics']['rmse']:.2f}")
    print(f"Chanel: R² = {chanel_results['test_metrics']['r2']:.3f}, RMSE = {chanel_results['test_metrics']['rmse']:.2f}")
    
    print("\n" + "="*60)
    print("COMPARISON: WEEKLY vs MONTHLY LSTM")
    print("="*60)
    print("\nMonthly LSTM (previous results):")
    print("  Zara:   R² = -2.331")
    print("  Chanel: R² = -1.614")
    print(f"\nWeekly LSTM with Google Trends (current):")
    print(f"  Zara:   R² = {zara_results['test_metrics']['r2']:.3f}")
    print(f"  Chanel: R² = {chanel_results['test_metrics']['r2']:.3f}")