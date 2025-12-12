"""
LSTM Model

This module implements LSTM (Long Short-Term Memory) for fashion trend prediction.

LSTM (Long Short-Term Memory) EXPLAINED:
-----------------------------------------

WHAT IS A RECURRENT NEURAL NETWORK (RNN)?
A neural network that processes sequences by maintaining a "hidden state"
that gets updated at each time step. Unlike feedforward networks, RNNs
have loops that allow information to persist.

THE PROBLEM WITH BASIC RNNs:
Vanishing gradient problem - when training on long sequences, gradients
become very small and the network "forgets" information from earlier
time steps. It's like playing telephone with 100 people - the message
gets lost.

HOW LSTM SOLVES THIS:
LSTM introduces "gates" that control information flow. Think of it as
having a notepad (cell state) where you can:
1. FORGET: Erase old notes that aren't relevant anymore
2. INPUT: Write new important information
3. OUTPUT: Read from your notes to make decisions

THE LSTM CELL (simplified):
----------------------------
                    Cell State (Long-term memory)
                           |
            +----[Forget]--+--[Input]----+
            |       |            |       |
            v       v            v       v
    Previous   Forget Gate   Input Gate   New
    Cell State  (0-1)         (0-1)       Info
            |       |            |        |
            +-------+------------+--------+
                           |
                    New Cell State
                           |
                    [Output Gate]
                           |
                    Hidden State (Short-term memory / Output)

GATE EQUATIONS:
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   # Forget gate
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)   # Input gate
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)   # Output gate
C_t = f_t * C_{t-1} + i_t * tanh(...)       # Cell state update
h_t = o_t * tanh(C_t)                        # Hidden state

WHERE:
- sigmoid outputs 0-1 (acts as a "gate" - 0=closed, 1=open)
- tanh outputs -1 to 1 (the actual information)

WHY LSTM FOR FASHION TRENDS?
- Naturally handles sequential data
- Can learn long-term dependencies (yearly patterns)
- Doesn't require manual lag features (learns them internally)
- Good at capturing complex temporal patterns

WHY GOOGLE TRENDS WITH LSTM?
- LSTM can learn the relationship between search interest and adoption
- Can discover if searches lead or lag actual behavior
- Adds external signal beyond just historical frequency
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

class LSTMModel(nn.Module):
    """
    LSTM Neural Network for time series prediction.
    
    ARCHITECTURE:
    -------------
    Input (sequence of features)
           |
           v
    LSTM Layer(s) - Process sequence, maintain memory
           |
           v
    Dropout - Prevent overfitting
           |
           v
    Dense Layer - Transform to prediction
           |
           v
    Output (predicted value)
    
    Parameters:
    -----------
    input_size : int
        Number of features per time step
    hidden_size : int
        Number of LSTM units (memory capacity)
        More = can learn more complex patterns, but slower
    num_layers : int
        Number of stacked LSTM layers
        More layers = deeper model, can learn hierarchical patterns
    dropout : float
        Dropout rate (0-1) for regularization
        Randomly "drops" connections during training to prevent overfitting
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
      
        # batch_first=True means input shape is (batch, sequence, features)
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
        """
        Forward pass through the network.
        
        HOW IT WORKS:
        1. LSTM processes the entire sequence
        2. We take the LAST hidden state (contains info from all time steps)
        3. Pass through dense layers to get prediction
        
        Input shape: (batch_size, sequence_length, input_size)
        Output shape: (batch_size,)
        """
        # lstm_out: all hidden states for each time step
        # (h_n, c_n): final hidden state and cell state
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        last_output = lstm_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out.squeeze()


def create_sequences(features: np.ndarray, 
                    target: np.ndarray, 
                    sequence_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    SEQUENCE CREATION EXPLAINED:
    ----------------------------
    LSTM needs data as sequences: "Given the past N time steps, predict the next value"
    
    Example with sequence_length=3:
    
    Original data:
        Time:     1    2    3    4    5    6
        Value:   10   20   30   40   50   60
    
    Created sequences:
        X[0] = [10, 20, 30] -> y[0] = 40  (use months 1-3 to predict month 4)
        X[1] = [20, 30, 40] -> y[1] = 50  (use months 2-4 to predict month 5)
        X[2] = [30, 40, 50] -> y[2] = 60  (use months 3-5 to predict month 6)
    
    This is a SLIDING WINDOW approach.
    
    CHOOSING SEQUENCE LENGTH:
    - Too short (e.g., 3): Might miss long-term patterns
    - Too long (e.g., 24): Less training samples, harder to train
    - 12 months: Captures yearly seasonality
    
    Parameters:
    -----------
    features : array
        Feature matrix (n_samples, n_features)
    target : array
        Target values (n_samples,)
    sequence_length : int
        Number of past time steps to use
    
    Returns:
    --------
    X : array of shape (n_sequences, sequence_length, n_features)
    y : array of shape (n_sequences,)
    """
    X, y = [], []
    
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    
    return np.array(X), np.array(y)


class LSTMForecaster:
    """
    LSTM model wrapper for fashion trend forecasting.
    
    This class handles:
    - Data preprocessing (scaling)
    - Sequence creation
    - Model training with PyTorch
    - Prediction
    - Evaluation
    """
    
    def __init__(self, sequence_length: int = 12, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 learning_rate: float = 0.001, epochs: int = 100,
                 batch_size: int = 16):
        """
        Initialize LSTM forecaster.
        
        HYPERPARAMETERS EXPLAINED:
        --------------------------
        sequence_length: How many past months to look at (12 = 1 year)
        
        hidden_size: LSTM memory capacity (64)
            - More = can learn complex patterns
            - Too many = overfitting, slow
        
        num_layers: Stacked LSTM layers (2)
            - More layers = deeper model
            - Usually 1-3 is enough
        
        dropout: Regularization strength (0.2 = drop 20%)
            - Higher = more regularization
            - Prevents overfitting
        
        learning_rate: Step size for optimization (0.001)
            - Too high = unstable training
            - Too low = slow convergence
        
        epochs: Number of training iterations (100)
            - One epoch = one pass through all data
        
        batch_size: Samples per gradient update (16)
            - Smaller = noisier but more updates
            - Larger = smoother but fewer updates
        """
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
        """
        Prepare data for LSTM: scale and create sequences.
        
        SCALING EXPLAINED:
        ------------------
        Neural networks work better when inputs are in similar ranges.
        MinMaxScaler transforms data to [0, 1] range:
        
        scaled = (value - min) / (max - min)
        
        Example:
            Original: [100, 200, 300, 400, 500]
            Scaled:   [0.0, 0.25, 0.5, 0.75, 1.0]
        
        We need to UNSCALE predictions back to original range:
            original = scaled * (max - min) + min
        
        IMPORTANT: We fit scalers on TRAINING data only!
        Using test data to fit scalers would be data leakage.
        """
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
        """
        Train the LSTM model.
        
        TRAINING PROCESS:
        -----------------
        1. Prepare data (scale, create sequences)
        2. Initialize model and optimizer
        3. For each epoch:
           a. Forward pass: compute predictions
           b. Compute loss (MSE between predictions and actual)
           c. Backward pass: compute gradients
           d. Update weights using optimizer
        4. Track loss history for monitoring
        
        LOSS FUNCTION (MSE):
        MSE = mean((predicted - actual)^2)
        Lower is better. We want to minimize this.
        
        OPTIMIZER (Adam):
        Adam = Adaptive Moment Estimation
        - Combines benefits of momentum and adaptive learning rates
        - Good default choice for most problems
        """
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
        """
        Make predictions with trained model.
        
        Note: Due to sequence creation, we can only predict for
        time points that have `sequence_length` prior observations.
        """
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
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance."""

        predictions = self.predict(X)
       
        y_actual = y.values[self.sequence_length:]
        
        rmse = np.sqrt(mean_squared_error(y_actual, predictions))
        mae = mean_absolute_error(y_actual, predictions)
        r2 = r2_score(y_actual, predictions)
        
        mask = y_actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_actual[mask] - predictions[mask]) / y_actual[mask])) * 100
        else:
            mape = np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'actual': y_actual
        }
    
    def plot_training_loss(self, save_path: str = None):
        """Plot training loss over epochs."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_losses, color='#457B9D', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('LSTM Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def train_and_evaluate_lstm(split_data: Dict, 
                           model_name: str = "LSTM",
                           sequence_length: int = 6,
                           epochs: int = 100) -> Dict:
    """
    Complete LSTM training and evaluation pipeline.
    
    Parameters:
    -----------
    split_data : dict
        Output from train_test_split
    model_name : str
        Name for display
    sequence_length : int
        LSTM sequence length (past months to consider)
    epochs : int
        Training epochs
    """
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
    print(f"Device: {device}")
    
    model = LSTMForecaster(
        sequence_length=3,
        hidden_size=16,
        num_layers=1,
        dropout=0.1,
        learning_rate=0.0005,
        epochs=epochs,
        batch_size=8
    )
    
    print("\nTraining LSTM...")
    model.train(X_train, y_train, X_test, y_test, verbose=True)
    
    #bridged data
    bridge_length = sequence_length
    X_bridge = pd.concat([X_train.iloc[-bridge_length:], X_test])
    y_bridge = pd.concat([y_train.iloc[-bridge_length:], y_test])
    
    
    predictions = model.predict(X_bridge)
    
    y_actual = y_test.values
    
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    mae = mean_absolute_error(y_actual, predictions)
    r2 = r2_score(y_actual, predictions)
    
    mask = y_actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_actual[mask] - predictions[mask]) / y_actual[mask])) * 100
    else:
        mape = np.nan
    
    test_metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': predictions,
        'actual': y_actual
    }
    
    print("\n--- Test Performance ---")
    test_metrics = model.evaluate(X_test, y_test)
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAE:  {test_metrics['mae']:.2f}")
    print(f"  R2:   {test_metrics['r2']:.3f}")
    print(f"  MAPE: {test_metrics['mape']:.1f}%")
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'predictions': predictions,
        'actual': y_actual,
        'test_dates': split_data['test_dates'],
        'train_dates': split_data['train_dates']
    }


def plot_lstm_predictions(results: Dict, 
                         train_y: pd.Series,
                         train_dates: pd.DatetimeIndex,
                         title: str = None,
                         save_path: str = None):
    """Visualize LSTM predictions vs actual."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(train_dates, train_y, color='#2A9D8F', 
            linewidth=2, label='Training (Actual)')
    
    ax.plot(results['test_dates'], results['actual'],
            color='#457B9D', linewidth=2, label='Test (Actual)')
    
    ax.plot(results['test_dates'], results['predictions'],
            color='#E63946', linewidth=2, linestyle='--',
            label='Test (Predicted)', marker='o', markersize=4)
   
    ax.axvline(x=train_dates[-1], color='black', 
               linestyle='--', alpha=0.5, label='Train/Test Split')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# ============================================================
# MAIN - Run this file to train LSTM models
# ============================================================
if __name__ == "__main__":
    import os
    from load_data import load_all_data
    from feature_eng import prepare_all_model_features
    from train_test import prepare_all_splits
    
    os.makedirs('plots', exist_ok=True)
    
    print(f"Using device: {device}")
    
    print("\nLoading data...")
    data = load_all_data(
        sfs_path='data/trend_counts_over_time.csv',
        google_path='data/google_trends.csv',
        weather_path='data/California_weather.csv'
    )

    print("\nCreating features...")
    feature_datasets = prepare_all_model_features(
        data['sfs'], data['google'], data['weather']
    )
  
    print("\nCreating train/test splits...")
    splits = prepare_all_splits(
        feature_datasets, data['sfs'],
        train_fraction=0.75, visualize=False, save_plots=False
    )
   
    zara_results = train_and_evaluate_lstm(
        splits['zara_lstm'],
        model_name="Zara LSTM (SFS + Google Trends)",
        sequence_length=3,
        epochs=100
    )
    
    plot_lstm_predictions(
        zara_results,
        splits['zara_lstm']['y_train'],
        splits['zara_lstm']['train_dates'],
        title='Zara Dress: LSTM Predictions vs Actual',
        save_path='plots/zara_lstm_predictions.png'
    )
    
    chanel_results = train_and_evaluate_lstm(
        splits['chanel_lstm'],
        model_name="Chanel LSTM (SFS + Google Trends)",
        sequence_length=3,
        epochs=100
    )
    
    plot_lstm_predictions(
        chanel_results,
        splits['chanel_lstm']['y_train'],
        splits['chanel_lstm']['train_dates'],
        title='Chanel Bag: LSTM Predictions vs Actual',
        save_path='plots/chanel_lstm_predictions.png'
    )
    
    print("\n" + "="*60)
    print("LSTM SUMMARY")
    print("="*60)
    print(f"\nZara:   Test RMSE={zara_results['test_metrics']['rmse']:.2f}, R2={zara_results['test_metrics']['r2']:.3f}")
    print(f"Chanel: Test RMSE={chanel_results['test_metrics']['rmse']:.2f}, R2={chanel_results['test_metrics']['r2']:.3f}")
    # === SFS-ONLY LSTM FOR COMPARISON ===
    print("\n" + "="*60)
    print("TRAINING LSTM WITH SFS DATA ONLY (FOR COMPARISON)")
    print("="*60)

    zara_results_sfs = train_and_evaluate_lstm(
        splits['zara_xgb'],
        model_name="Zara LSTM (SFS only)",
        sequence_length=3,
        epochs=100
        )

    plot_lstm_predictions(
        zara_results_sfs,
        splits['zara_xgb']['y_train'],
        splits['zara_xgb']['train_dates'],
        title='Zara Dress: LSTM (SFS only) Predictions vs Actual',
        save_path='plots/zara_lstm_sfs_only_predictions.png'
        )

    chanel_results_sfs = train_and_evaluate_lstm(
        splits['chanel_xgb'],
        model_name="Chanel LSTM (SFS only)",
        sequence_length=3,
        epochs=100
        )

    plot_lstm_predictions(
        chanel_results_sfs,
        splits['chanel_xgb']['y_train'],
        splits['chanel_xgb']['train_dates'],
        title='Chanel Bag: LSTM (SFS only) Predictions vs Actual',
        save_path='plots/chanel_lstm_sfs_only_predictions.png'
        )

    print("\n" + "="*60)
    print("LSTM COMPARISON SUMMARY")
    print("="*60)
    print("\nWith Google Trends:")
    print(f"  Zara:   RMSE={zara_results['test_metrics']['rmse']:.2f}, R2={zara_results['test_metrics']['r2']:.3f}")
    print(f"  Chanel: RMSE={chanel_results['test_metrics']['rmse']:.2f}, R2={chanel_results['test_metrics']['r2']:.3f}")
    print("\nSFS Only:")
    print(f"  Zara:   RMSE={zara_results_sfs['test_metrics']['rmse']:.2f}, R2={zara_results_sfs['test_metrics']['r2']:.3f}")
    print(f"  Chanel: RMSE={chanel_results_sfs['test_metrics']['rmse']:.2f}, R2={chanel_results_sfs['test_metrics']['r2']:.3f}")