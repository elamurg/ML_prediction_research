"""
TFT Model - Weekly Data Version

This module implements Temporal Fusion Transformer for fashion trend prediction
using WEEKLY data with Google Trends.
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


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network - learns which features are important."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.feature_transform = nn.Linear(input_size, hidden_size)
        self.weight_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        weights = self.weight_network(x)
        weighted_x = x * weights
        output = self.feature_transform(weighted_x)
        output = self.dropout(output)
        return output, weights


class TemporalAttention(nn.Module):
    """Temporal Self-Attention."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output(attended)
        
        avg_attention = attention_weights.mean(dim=1)
        return output, avg_attention


class SimplifiedTFT(nn.Module):
    """Simplified Temporal Fusion Transformer."""
    
    def __init__(self, input_size: int, hidden_size: int = 64,
                 lstm_layers: int = 1, attention_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.var_selection = VariableSelectionNetwork(input_size, hidden_size, dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True,
                           dropout=dropout if lstm_layers > 1 else 0)
        self.attention = TemporalAttention(hidden_size, attention_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
        self.feature_weights = None
        self.attention_weights = None
    
    def forward(self, x):
        selected, feature_weights = self.var_selection(x)
        self.feature_weights = feature_weights
        
        lstm_out, _ = self.lstm(selected)
        lstm_out = self.layer_norm1(lstm_out + selected)
        
        attended, attention_weights = self.attention(lstm_out)
        self.attention_weights = attention_weights
        attended = self.layer_norm2(attended + lstm_out)
        
        final = attended[:, -1, :]
        out = self.dropout(final)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out.squeeze()


class TFTForecaster:
    """TFT wrapper for fashion trend forecasting."""
    
    def __init__(self, sequence_length: int = 12, hidden_size: int = 64,
                 lstm_layers: int = 1, attention_heads: int = 4,
                 dropout: float = 0.2, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 16):
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.training_losses = []
        self.feature_names = None
    
    def create_sequences(self, features: np.ndarray, target: np.ndarray):
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, fit_scalers: bool = True):
        X_array = X.values
        y_array = y.values.reshape(-1, 1)
        
        if fit_scalers:
            X_scaled = self.feature_scaler.fit_transform(X_array)
            y_scaled = self.target_scaler.fit_transform(y_array).flatten()
        else:
            X_scaled = self.feature_scaler.transform(X_array)
            y_scaled = self.target_scaler.transform(y_array).flatten()
        
        return self.create_sequences(X_scaled, y_scaled)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              verbose: bool = True):
        
        self.feature_names = list(X_train.columns)
        X_train_seq, y_train_seq = self.prepare_data(X_train, y_train, fit_scalers=True)
        
        if X_val is not None:
            X_val_seq, y_val_seq = self.prepare_data(X_val, y_val, fit_scalers=False)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
            y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
        
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = SimplifiedTFT(
            input_size=X_train.shape[1],
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            attention_heads=self.attention_heads,
            dropout=self.dropout
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        
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
            
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor)
                    val_loss = criterion(val_pred, y_val_tensor).item()
                scheduler.step(val_loss)
            else:
                scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
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
        X_seq, _ = self.create_sequences(X_scaled, dummy_y)
        
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        return self.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")
        
        X_scaled = self.feature_scaler.transform(X.values)
        dummy_y = np.zeros(len(X))
        X_seq, _ = self.create_sequences(X_scaled, dummy_y)
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(X_tensor)
            weights = self.model.feature_weights.cpu().numpy()
        
        avg_weights = weights.mean(axis=(0, 1))
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_weights
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    def plot_feature_importance(self, X: pd.DataFrame, top_n: int = 15,
                                title: str = None, save_path: str = None):
        importance_df = self.get_feature_importance(X)
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), top_features['importance'].values[::-1], color='#E76F51')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['feature'].values[::-1])
        ax.set_xlabel('Feature Importance')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return importance_df


# Import feature preparation from lstm_weekly
from lstm_weekly import prepare_weekly_features, create_weekly_train_test_split


def train_and_evaluate_tft_weekly(split_data: Dict,
                                  model_name: str = "TFT",
                                  sequence_length: int = 12,
                                  hidden_size: int = 64,
                                  attention_heads: int = 4,
                                  epochs: int = 100) -> Dict:
    """Complete TFT training and evaluation."""
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
    
    model = TFTForecaster(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        lstm_layers=1,
        attention_heads=attention_heads,
        dropout=0.2,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=32
    )
    
    print("\nTraining TFT...")
    model.train(X_train, y_train, X_test, y_test, verbose=True)
    
    # Bridge for predictions
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
        'train_dates': split_data['train_dates'],
        'X_test': X_test
    }


def plot_tft_predictions_weekly(results: Dict, 
                                train_y: pd.Series,
                                train_dates: pd.DatetimeIndex,
                                title: str = None,
                                save_path: str = None):
    """Visualize TFT predictions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(train_dates, train_y, color='#2A9D8F', linewidth=1.5, label='Training', alpha=0.8)
    ax.plot(results['test_dates'], results['actual'], color='#457B9D', linewidth=1.5, label='Test (Actual)')
    ax.plot(results['test_dates'], results['predictions'], color='#E63946', linewidth=1.5, 
            linestyle='--', label='Test (Predicted)', alpha=0.8)
    ax.axvline(x=train_dates[-1], color='black', linestyle='--', alpha=0.5)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    metrics = results['test_metrics']
    ax.annotate(f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.2f}", 
                xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, verticalalignment='top',
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
    
    # Load data
    data = load_all_weekly_data(
        sfs_metadata_path='data/SFS_metadata.csv',
        weather_path='data/California_weather.csv',
        google_trends_path='data/google_trends_weekly_smoothed.csv'
    )
    
    # === ZARA ===
    print("\n" + "="*60)
    print("ZARA DRESS - WEEKLY TFT")
    print("="*60)
    
    zara_features = prepare_weekly_features(data['sfs'], data['weather'], data['google'], 'zara_frequency')
    zara_split = create_weekly_train_test_split(zara_features, 'zara_frequency')
    
    zara_results = train_and_evaluate_tft_weekly(
        zara_split, model_name="Zara TFT (Weekly)", sequence_length=12, epochs=150
    )
    
    plot_tft_predictions_weekly(
        zara_results, zara_split['y_train'], zara_split['train_dates'],
        title='Zara Dress: Weekly TFT Predictions', save_path='plots/zara_tft_weeklyV2.png'
    )
    
    # Feature importance
    print("\n--- Feature Importance ---")
    importance = zara_results['model'].get_feature_importance(zara_results['X_test'])
    print(importance.head(15))
    
    zara_results['model'].plot_feature_importance(
        zara_results['X_test'], top_n=15,
        title='Zara: TFT Feature Importance', save_path='plots/zara_tft_importance.png'
    )
    
    # === CHANEL ===
    print("\n" + "="*60)
    print("CHANEL BAG - WEEKLY TFT")
    print("="*60)
    
    chanel_features = prepare_weekly_features(data['sfs'], data['weather'], data['google'], 'chanel_frequency')
    chanel_split = create_weekly_train_test_split(chanel_features, 'chanel_frequency')
    
    chanel_results = train_and_evaluate_tft_weekly(
        chanel_split, model_name="Chanel TFT (Weekly)", sequence_length=12, epochs=150
    )
    
    plot_tft_predictions_weekly(
        chanel_results, chanel_split['y_train'], chanel_split['train_dates'],
        title='Chanel Bag: Weekly TFT Predictions', save_path='plots/chanel_tft_weeklyV2.png'
    )
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("WEEKLY TFT SUMMARY")
    print("="*60)
    print(f"\nZara:   R² = {zara_results['test_metrics']['r2']:.3f}, RMSE = {zara_results['test_metrics']['rmse']:.2f}")
    print(f"Chanel: R² = {chanel_results['test_metrics']['r2']:.3f}, RMSE = {chanel_results['test_metrics']['rmse']:.2f}")