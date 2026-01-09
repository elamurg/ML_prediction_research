"""
Stage 6: Model Comparison and Evaluation (SFS-Only Models)
==========================================================

This module compares all three SFS-only models and provides comprehensive evaluation.

MODELS COMPARED:
- XGBoost (SFS Only): Gradient boosting baseline
- LSTM (SFS Only): Recurrent neural network
- TFT (SFS Only): Temporal Fusion Transformer

EVALUATION METRICS:
- RMSE: Root Mean Squared Error (lower is better)
- MAE: Mean Absolute Error (lower is better)
- R²: Coefficient of Determination (higher is better, max 1.0)
- MAPE: Mean Absolute Percentage Error (lower is better)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create a comparison table of all model results."""
    rows = []
    for model_name, model_results in results.items():
        metrics = model_results['test_metrics']
        rows.append({
            'Model': model_name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R2': metrics['r2'],
            'MAPE': metrics['mape']
        })
    df = pd.DataFrame(rows)
    df = df.set_index('Model')
    return df


def plot_all_predictions(results: Dict, title: str, save_path: str = None):
    """Plot predictions from all models on the same chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {
        'XGBoost': '#2A9D8F',
        'LSTM': '#E76F51', 
        'TFT': '#457B9D'
    }
    
    # Get first model's data for actual values
    first_model = list(results.keys())[0]
    first_result = results[first_model]
    
    # Plot training data (from any model - they should be the same)
    if 'y_train' in first_result:
        ax.plot(first_result['train_dates'], first_result['y_train'], 
                color='gray', linewidth=1.5, label='Training (Actual)', alpha=0.6)
    
    # Plot actual test data
    ax.plot(first_result['test_dates'], first_result['actual'], 
            color='black', linewidth=2, label='Test (Actual)')
    
    # Plot predictions from each model
    for model_name, model_results in results.items():
        color = colors.get(model_name.split()[0], '#666666')
        r2 = model_results['test_metrics']['r2']
        ax.plot(model_results['test_dates'], model_results['predictions'],
                linewidth=1.5, linestyle='--', color=color, alpha=0.8,
                label=f'{model_name} (R²={r2:.3f})')
    
    # Add train/test split line
    if 'train_dates' in first_result:
        ax.axvline(x=first_result['train_dates'][-1], color='black', 
                   linestyle=':', alpha=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Frequency')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def create_summary_visualization(zara_results: Dict, chanel_results: Dict,
                                save_path: str = None):
    """Create comprehensive summary visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['rmse', 'r2']
    metric_labels = ['RMSE (Lower is Better)', 'R² (Higher is Better)']
    colors = ['#2A9D8F', '#E76F51', '#457B9D']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Zara
        ax1 = axes[idx, 0]
        zara_models = list(zara_results.keys())
        zara_values = [zara_results[m]['test_metrics'][metric] for m in zara_models]
        
        bars1 = ax1.bar(range(len(zara_models)), zara_values, color=colors[:len(zara_models)])
        ax1.set_xticks(range(len(zara_models)))
        ax1.set_xticklabels([m.replace(' (SFS Only)', '\n(SFS)') for m in zara_models], fontsize=9)
        ax1.set_ylabel(label)
        ax1.set_title(f'Zara Dress - {label}', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, zara_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Chanel
        ax2 = axes[idx, 1]
        chanel_models = list(chanel_results.keys())
        chanel_values = [chanel_results[m]['test_metrics'][metric] for m in chanel_models]
        
        bars2 = ax2.bar(range(len(chanel_models)), chanel_values, color=colors[:len(chanel_models)])
        ax2.set_xticks(range(len(chanel_models)))
        ax2.set_xticklabels([m.replace(' (SFS Only)', '\n(SFS)') for m in chanel_models], fontsize=9)
        ax2.set_ylabel(label)
        ax2.set_title(f'Chanel Bag - {label}', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, chanel_values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Comparison (SFS Data Only)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def print_analysis_summary(zara_comparison: pd.DataFrame, chanel_comparison: pd.DataFrame):
    """Print comprehensive analysis summary."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS (SFS-ONLY MODELS)")
    print("="*70)
    
    print("\n" + "-"*70)
    print("ZARA DRESS RESULTS")
    print("-"*70)
    print(zara_comparison.round(3).to_string())
    
    best_rmse_zara = zara_comparison['RMSE'].idxmin()
    best_r2_zara = zara_comparison['R2'].idxmax()
    print(f"\n  Best RMSE: {best_rmse_zara} ({zara_comparison.loc[best_rmse_zara, 'RMSE']:.3f})")
    print(f"  Best R²:   {best_r2_zara} ({zara_comparison.loc[best_r2_zara, 'R2']:.3f})")
    
    print("\n" + "-"*70)
    print("CHANEL BAG RESULTS")
    print("-"*70)
    print(chanel_comparison.round(3).to_string())
    
    best_rmse_chanel = chanel_comparison['RMSE'].idxmin()
    best_r2_chanel = chanel_comparison['R2'].idxmax()
    print(f"\n  Best RMSE: {best_rmse_chanel} ({chanel_comparison.loc[best_rmse_chanel, 'RMSE']:.3f})")
    print(f"  Best R²:   {best_r2_chanel} ({chanel_comparison.loc[best_r2_chanel, 'R2']:.3f})")
    
    print("\n" + "-"*70)
    print("KEY FINDINGS")
    print("-"*70)
    
    # Analyze results
    zara_xgb_r2 = zara_comparison.loc['XGBoost (SFS Only)', 'R2'] if 'XGBoost (SFS Only)' in zara_comparison.index else None
    zara_lstm_r2 = zara_comparison.loc['LSTM (SFS Only)', 'R2'] if 'LSTM (SFS Only)' in zara_comparison.index else None
    zara_tft_r2 = zara_comparison.loc['TFT (SFS Only)', 'R2'] if 'TFT (SFS Only)' in zara_comparison.index else None
    
    print("\n  1. XGBoost vs Neural Networks (LSTM/TFT):")
    if zara_xgb_r2 and zara_lstm_r2 and zara_tft_r2:
        best_nn = max(zara_lstm_r2, zara_tft_r2)
        if zara_xgb_r2 > best_nn:
            print(f"     XGBoost outperforms neural networks on this dataset.")
            print(f"     This is expected with ~400 samples - tree-based methods handle small data better.")
        else:
            print(f"     Neural networks match or beat XGBoost!")
            print(f"     The simplified architecture helped avoid overfitting.")
    
    print("\n  2. LSTM vs TFT:")
    if zara_lstm_r2 and zara_tft_r2:
        if zara_tft_r2 > zara_lstm_r2:
            print(f"     TFT outperforms LSTM - attention mechanism helps.")
        elif zara_lstm_r2 > zara_tft_r2:
            print(f"     LSTM outperforms TFT - simpler model works better here.")
        else:
            print(f"     LSTM and TFT perform similarly.")
    
    print("\n  3. Dataset Size Impact:")
    print(f"     With ~400 weekly samples, deep learning models face challenges.")
    print(f"     Removing external features (weather, Google Trends) reduces noise.")
    
    print("\n" + "-"*70)
    print("R² INTERPRETATION GUIDE")
    print("-"*70)
    print("""
    R² > 0.8:  Excellent - model explains most variance
    R² 0.6-0.8: Good - model captures main patterns  
    R² 0.4-0.6: Moderate - model has predictive value
    R² 0.2-0.4: Weak - limited predictive power
    R² < 0.2:  Poor - model struggles to predict
    R² < 0:    Very Poor - worse than predicting the mean
    """)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import os
    
    # Import SFS-only models
    from xgb_weekly_SFS_only import (
        prepare_sfs_only_features as prepare_xgb_features,
        create_train_test_split as create_xgb_split,
        train_and_evaluate_xgboost
    )
    from lstm_weekly_SFS_only import (
        prepare_sfs_only_features as prepare_lstm_features,
        create_train_test_split as create_lstm_split,
        train_and_evaluate_lstm
    )
    from tft_weekly_SFS_only import (
        prepare_sfs_only_features as prepare_tft_features,
        create_train_test_split as create_tft_split,
        train_and_evaluate_tft
    )
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    sfs_df = pd.read_csv('data/weekly_trend_counts.csv', parse_dates=['date'], index_col='date')
    sfs_df['zara_frequency'] = sfs_df['zara_dress_interp']
    sfs_df['chanel_frequency'] = sfs_df['chanel_bag_interp']
    print(f"Loaded {len(sfs_df)} weeks of data")
    
    # ================================================================
    # ZARA MODELS
    # ================================================================
    print("\n" + "="*70)
    print("TRAINING ZARA MODELS (SFS ONLY)")
    print("="*70)
    
    # XGBoost
    zara_xgb_features = prepare_xgb_features(sfs_df, 'zara_frequency')
    zara_xgb_split = create_xgb_split(zara_xgb_features, 'zara_frequency')
    zara_xgb = train_and_evaluate_xgboost(zara_xgb_split, "Zara XGBoost (SFS Only)")
    zara_xgb['y_train'] = zara_xgb_split['y_train']
    zara_xgb['actual'] = zara_xgb_split['y_test'].values
    
    # LSTM
    zara_lstm_features = prepare_lstm_features(sfs_df, 'zara_frequency')
    zara_lstm_split = create_lstm_split(zara_lstm_features, 'zara_frequency')
    zara_lstm = train_and_evaluate_lstm(zara_lstm_split, "Zara LSTM (SFS Only)", 
                                        sequence_length=8, hidden_size=32, epochs=150)
    
    # TFT
    zara_tft_features = prepare_tft_features(sfs_df, 'zara_frequency')
    zara_tft_split = create_tft_split(zara_tft_features, 'zara_frequency')
    zara_tft = train_and_evaluate_tft(zara_tft_split, "Zara TFT (SFS Only)",
                                      sequence_length=8, hidden_size=32, epochs=150)
    
    zara_results = {
        'XGBoost (SFS Only)': zara_xgb,
        'LSTM (SFS Only)': zara_lstm,
        'TFT (SFS Only)': zara_tft
    }
    
    # ================================================================
    # CHANEL MODELS
    # ================================================================
    print("\n" + "="*70)
    print("TRAINING CHANEL MODELS (SFS ONLY)")
    print("="*70)
    
    # XGBoost
    chanel_xgb_features = prepare_xgb_features(sfs_df, 'chanel_frequency')
    chanel_xgb_split = create_xgb_split(chanel_xgb_features, 'chanel_frequency')
    chanel_xgb = train_and_evaluate_xgboost(chanel_xgb_split, "Chanel XGBoost (SFS Only)")
    chanel_xgb['y_train'] = chanel_xgb_split['y_train']
    chanel_xgb['actual'] = chanel_xgb_split['y_test'].values
    
    # LSTM
    chanel_lstm_features = prepare_lstm_features(sfs_df, 'chanel_frequency')
    chanel_lstm_split = create_lstm_split(chanel_lstm_features, 'chanel_frequency')
    chanel_lstm = train_and_evaluate_lstm(chanel_lstm_split, "Chanel LSTM (SFS Only)",
                                          sequence_length=8, hidden_size=32, epochs=150)
    
    # TFT
    chanel_tft_features = prepare_tft_features(sfs_df, 'chanel_frequency')
    chanel_tft_split = create_tft_split(chanel_tft_features, 'chanel_frequency')
    chanel_tft = train_and_evaluate_tft(chanel_tft_split, "Chanel TFT (SFS Only)",
                                        sequence_length=8, hidden_size=32, epochs=150)
    
    chanel_results = {
        'XGBoost (SFS Only)': chanel_xgb,
        'LSTM (SFS Only)': chanel_lstm,
        'TFT (SFS Only)': chanel_tft
    }
    
    # ================================================================
    # COMPARISON AND VISUALIZATION
    # ================================================================
    print("\n" + "="*70)
    print("GENERATING COMPARISON RESULTS")
    print("="*70)
    
    # Create comparison tables
    zara_comparison = create_comparison_table(zara_results)
    chanel_comparison = create_comparison_table(chanel_results)
    
    # Print analysis
    print_analysis_summary(zara_comparison, chanel_comparison)
    
    # Plot all predictions together
    plot_all_predictions(zara_results, 
                        'Zara Dress: All Models Comparison (SFS Only)',
                        'plots/zara_all_models_comparison.png')
    
    plot_all_predictions(chanel_results,
                        'Chanel Bag: All Models Comparison (SFS Only)', 
                        'plots/chanel_all_models_comparison.png')
    
    # Create summary bar charts
    create_summary_visualization(zara_results, chanel_results, 
                                'plots/model_comparison_sfs_only.png')
    
    # Save results to CSV
    zara_comparison.to_csv('results/zara_comparison_sfs_only.csv')
    chanel_comparison.to_csv('results/chanel_comparison_sfs_only.csv')
    
    print("\n" + "="*70)
    print("RESULTS SAVED")
    print("="*70)
    print("  - plots/zara_all_models_comparison.png")
    print("  - plots/chanel_all_models_comparison.png")
    print("  - plots/model_comparison_sfs_only.png")
    print("  - results/zara_comparison_sfs_only.csv")
    print("  - results/chanel_comparison_sfs_only.csv")