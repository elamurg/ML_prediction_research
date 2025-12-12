"""
Stage 5C: Temporal Fusion Transformer (TFT) Model
==================================================

This module implements a simplified TFT for fashion trend prediction.

TFT (Temporal Fusion Transformer) EXPLAINED:
---------------------------------------------

WHAT IS A TRANSFORMER?
The Transformer architecture, introduced in "Attention Is All You Need" (2017),
revolutionized sequence modeling by using ATTENTION instead of recurrence.

ATTENTION MECHANISM (simplified):
---------------------------------
Attention answers: "When predicting for time T, which past time steps matter most?"

Traditional approach (LSTM): Process sequentially, hope important info persists
Attention approach: Look at ALL time steps and learn which ones are relevant

Example for fashion trends:
- Predicting March 2014 frequency
- Attention might learn: "March 2013 is very relevant (seasonal pattern)"
- While "Random month in 2010" gets low attention weight

THE MATH (simplified):
Query (Q): "What am I looking for?"
Key (K): "What does each time step contain?"
Value (V): "What information does each time step have?"

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

- QK^T: How relevant is each key to my query?
- softmax: Convert to probabilities (sum to 1)
- Multiply by V: Weighted sum of values

WHY TFT FOR FASHION TRENDS?
1. INTERPRETABILITY: Attention weights show which features/time steps matter
2. MULTIPLE INPUTS: Handles different types of features naturally
3. LONG-RANGE DEPENDENCIES: Directly attends to distant past
4. STATE-OF-THE-ART: Often best performance on time series tasks

TFT COMPONENTS:
---------------
1. Variable Selection: Learn which features are important
2. LSTM Encoder: Process local temporal patterns
3. Attention Layer: Learn long-range dependencies
4. Dense Layers: Produce final prediction

WHY ALL THREE DATASETS FOR TFT?
- TFT can handle heterogeneous features well
- Attention learns which features matter for each prediction
- Weather adds seasonality context
- Google Trends adds consumer intent signal
- Model can discover complex interactions

This is a SIMPLIFIED version of TFT. The full architecture has more
components (gating mechanisms, multi-head attention, etc.) but the core
concepts remain the same.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network - learns which features are important.
    
    This component helps the model focus on relevant features and
    provides interpretability by showing feature importance.
    
    HOW IT WORKS:
    1. Takes all features as input
    2. Learns weights for each feature (via softmax)
    3. Outputs weighted combination
    
    This allows the model to automatically discover that, e.g.,
    "temperature matters a lot in summer" or "search interest matters
    more than humidity".
    """
    
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
        """
        x shape: (batch, seq_len, input_size)
        returns: (batch, seq_len, hidden_size), weights
        """
        weights = self.weight_network(x)

        weighted_x = x * weights

        output = self.feature_transform(weighted_x)
        output = self.dropout(output)
        
        return output, weights


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention - learns which time steps are important.
    
    SELF-ATTENTION EXPLAINED:
    -------------------------
    Each time step "attends" to all other time steps (including itself).
    
    For predicting the future:
    - Query: "What am I trying to predict?"
    - Keys: "What patterns exist at each past time step?"
    - Values: "What information should I extract?"
    
    The attention weights show temporal dependencies:
    - High weight on t-12: Strong yearly seasonality
    - High weight on t-1: Strong momentum/autocorrelation
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        """
        x shape: (batch, seq_len, hidden_size)
        returns: (batch, seq_len, hidden_size), attention_weights
        """
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x)  # (batch, seq, hidden)
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
    """
    Simplified Temporal Fusion Transformer.
    
    ARCHITECTURE:
    -------------
    Input Features
         |
         v
    Variable Selection -----> Feature Importance Weights
         |
         v
    LSTM Encoder (local patterns)
         |
         v
    Self-Attention (global patterns) -----> Temporal Attention Weights
         |
         v
    Dense Layers
         |
         v
    Prediction
    
    The model provides two types of interpretability:
    1. Feature importance (from Variable Selection)
    2. Temporal importance (from Attention weights)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64,
                 lstm_layers: int = 1, attention_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
  
        self.var_selection = VariableSelectionNetwork(input_size, hidden_size, dropout)
      
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
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
        """
        Forward pass with interpretability outputs.
        
        x shape: (batch, seq_len, input_size)
        returns: predictions (batch,)
        """
        selected, feature_weights = self.var_selection(x)
        self.feature_weights = feature_weights
  
        lstm_out, _ = self.lstm(selected)
        lstm_out = self.layer_norm1(lstm_out + selected)  # Residual connection
       
        attended, attention_weights = self.attention(lstm_out)
        self.attention_weights = attention_weights
        attended = self.layer_norm2(attended + lstm_out)  # Residual connection

        final = attended[:, -1, :]
   
        out = self.dropout(final)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out.squeeze()

class TFTForecaster:
    """
    TFT wrapper for fashion trend forecasting.
    
    Handles data preparation, training, prediction, and interpretability.
    """
    
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
    
    def create_sequences(self, features: np.ndarray, 
                        target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for TFT input."""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series,
                    fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale data for TFT."""
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
              verbose: bool = True) -> 'TFTForecaster':
        """Train the TFT model."""
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

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
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        X_scaled = self.feature_scaler.transform(X.values)
        dummy_y = np.zeros(len(X))
        X_seq, _ = self.create_sequences(X_scaled, dummy_y)
        
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance."""
        predictions = self.predict(X)
        y_actual = y.values[self.sequence_length:]
        
        rmse = np.sqrt(mean_squared_error(y_actual, predictions))
        mae = mean_absolute_error(y_actual, predictions)
        r2 = r2_score(y_actual, predictions)
        
        mask = y_actual != 0
        mape = np.mean(np.abs((y_actual[mask] - predictions[mask]) / y_actual[mask])) * 100 if mask.sum() > 0 else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'actual': y_actual
        }
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance from Variable Selection Network.
        
        Returns average attention weights for each feature.
        """
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
        """Visualize feature importance."""
        importance_df = self.get_feature_importance(X)
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), top_features['importance'].values[::-1], color='#E76F51')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['feature'].values[::-1])
        ax.set_xlabel('Feature Importance (Attention Weight)')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return importance_df

def train_and_evaluate_tft(split_data: Dict,
                          model_name: str = "TFT",
                          sequence_length: int = 6,
                          epochs: int = 100) -> Dict:
    """Complete TFT training and evaluation pipeline."""
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
    print(f"Sequence length: {sequence_length}")
    
    model = TFTForecaster(
        sequence_length=sequence_length,
        hidden_size=64,
        lstm_layers=1,
        attention_heads=4,
        dropout=0.2,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=16
    )
    
    print("\nTraining TFT...")
    model.train(X_train, y_train, X_test, y_test, verbose=True)
    
    print("\n--- Test Performance ---")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAE:  {test_metrics['mae']:.2f}")
    print(f"  R2:   {test_metrics['r2']:.3f}")
    print(f"  MAPE: {test_metrics['mape']:.1f}%")
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'predictions': test_metrics['predictions'],
        'actual': test_metrics['actual'],
        'test_dates': split_data['test_dates'][sequence_length:],
        'X_test': X_test
    }

def plot_tft_predictions(results: Dict, train_y: pd.Series,
                        train_dates: pd.DatetimeIndex,
                        title: str = None, save_path: str = None):
    """Visualize TFT predictions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(train_dates, train_y, color='#2A9D8F', linewidth=2, label='Training (Actual)')
    ax.plot(results['test_dates'], results['actual'], color='#457B9D', linewidth=2, label='Test (Actual)')
    ax.plot(results['test_dates'], results['predictions'], color='#E63946',
            linewidth=2, linestyle='--', label='Test (Predicted)', marker='o', markersize=4)
    
    ax.axvline(x=train_dates[-1], color='black', linestyle='--', alpha=0.5, label='Train/Test Split')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import os
    from load_data import load_all_data
    from feature_eng import prepare_all_model_features
    from train_test import prepare_all_splits
    
    os.makedirs('plots', exist_ok=True)
    print(f"Using device: {device}")
 
    data = load_all_data(
        sfs_path='data/trend_counts_over_time.csv',
        google_path='data/google_trends.csv',
        weather_path='data/California_weather.csv'
    )
    
    feature_datasets = prepare_all_model_features(data['sfs'], data['google'], data['weather'])
    splits = prepare_all_splits(feature_datasets, data['sfs'], train_fraction=0.75, visualize=False)

    zara_results = train_and_evaluate_tft(
        splits['zara_tft'],
        model_name="Zara TFT (SFS + Google + Weather)",
        sequence_length=6, epochs=100
    )
    
    plot_tft_predictions(
        zara_results, splits['zara_tft']['y_train'], splits['zara_tft']['train_dates'],
        title='Zara Dress: TFT Predictions vs Actual', save_path='plots/zara_tft_predictions.png'
    )
    
    zara_results['model'].plot_feature_importance(
        zara_results['X_test'], top_n=15,
        title='Zara Dress: TFT Feature Importance', save_path='plots/zara_tft_importance.png'
    )
    
    chanel_results = train_and_evaluate_tft(
        splits['chanel_tft'],
        model_name="Chanel TFT (SFS + Google + Weather)",
        sequence_length=6, epochs=100
    )
    
    plot_tft_predictions(
        chanel_results, splits['chanel_tft']['y_train'], splits['chanel_tft']['train_dates'],
        title='Chanel Bag: TFT Predictions vs Actual', save_path='plots/chanel_tft_predictions.png'
    )
    
    print("\n" + "="*60)
    print("TFT SUMMARY")
    print("="*60)
    print(f"\nZara:   Test RMSE={zara_results['test_metrics']['rmse']:.2f}, R2={zara_results['test_metrics']['r2']:.3f}")
    print(f"Chanel: Test RMSE={chanel_results['test_metrics']['rmse']:.2f}, R2={chanel_results['test_metrics']['r2']:.3f}")