"""
Stage 6: Model Comparison and Evaluation
=========================================

This module compares all three models and provides comprehensive evaluation.

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


def create_summary_visualization(zara_results: Dict, chanel_results: Dict,
                                save_path: str = None):
    """Create comprehensive summary visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['rmse', 'r2']
    metric_labels = ['RMSE (Lower is Better)', 'R2 (Higher is Better)']
    colors = ['#2A9D8F', '#E76F51', '#457B9D']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        # Zara
        ax1 = axes[idx, 0]
        zara_models = list(zara_results.keys())
        zara_values = [zara_results[m]['test_metrics'][metric] for m in zara_models]
        
        bars1 = ax1.bar(range(len(zara_models)), zara_values, color=colors)
        ax1.set_xticks(range(len(zara_models)))
        ax1.set_xticklabels(['XGBoost\n(SFS)', 'LSTM\n(SFS+Google)', 'TFT\n(All Data)'])
        ax1.set_ylabel(label)
        ax1.set_title(f'Zara Dress - {label}', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, zara_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Chanel
        ax2 = axes[idx, 1]
        chanel_models = list(chanel_results.keys())
        chanel_values = [chanel_results[m]['test_metrics'][metric] for m in chanel_models]
        
        bars2 = ax2.bar(range(len(chanel_models)), chanel_values, color=colors)
        ax2.set_xticks(range(len(chanel_models)))
        ax2.set_xticklabels(['XGBoost\n(SFS)', 'LSTM\n(SFS+Google)', 'TFT\n(All Data)'])
        ax2.set_ylabel(label)
        ax2.set_title(f'Chanel Bag - {label}', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, chanel_values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_analysis_summary(zara_comparison: pd.DataFrame, chanel_comparison: pd.DataFrame):
    """Print comprehensive analysis summary."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("="*70)
    
    print("\n--- ZARA DRESS RESULTS ---")
    print(zara_comparison.round(3).to_string())
    
    best_rmse_zara = zara_comparison['RMSE'].idxmin()
    best_r2_zara = zara_comparison['R2'].idxmax()
    print(f"\n  Best RMSE: {best_rmse_zara}")
    print(f"  Best R2:   {best_r2_zara}")
    
    print("\n--- CHANEL BAG RESULTS ---")
    print(chanel_comparison.round(3).to_string())
    
    best_rmse_chanel = chanel_comparison['RMSE'].idxmin()
    best_r2_chanel = chanel_comparison['R2'].idxmax()
    print(f"\n  Best RMSE: {best_rmse_chanel}")
    print(f"  Best R2:   {best_r2_chanel}")
    
    print("\n" + "-"*70)
    print("INTERPRETATION GUIDE")
    print("-"*70)
    print("""
    Q: Does Google Trends help predict fashion trends?
       Compare LSTM vs XGBoost - if LSTM is better, answer is YES
    
    Q: Does weather data help?
       Compare TFT vs LSTM - if TFT is better, answer is YES
    
    R2 Interpretation:
       > 0.7: Good fit
       0.4-0.7: Moderate fit
       < 0.4: Poor fit
    """)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import os
    from load_data import load_all_data
    from feature_eng import prepare_all_model_features
    from train_test import prepare_all_splits
    from xgboost import train_and_evaluate_xgboost
    from lstm import train_and_evaluate_lstm
    from tft import train_and_evaluate_tft
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and prepare data
    data = load_all_data('trend_counts_over_time.csv', 'google_trends.csv', 'California_weather.csv')
    feature_datasets = prepare_all_model_features(data['sfs'], data['google'], data['weather'])
    splits = prepare_all_splits(feature_datasets, data['sfs'], train_fraction=0.75, visualize=False)
    
    # Train all Zara models
    print("\n" + "="*70)
    print("TRAINING ZARA MODELS")
    print("="*70)
    zara_xgb = train_and_evaluate_xgboost(splits['zara_xgb'], "Zara XGBoost")
    zara_lstm = train_and_evaluate_lstm(splits['zara_lstm'], "Zara LSTM", sequence_length=6, epochs=100)
    zara_tft = train_and_evaluate_tft(splits['zara_tft'], "Zara TFT", sequence_length=6, epochs=100)
    
    zara_results = {
        'XGBoost (SFS)': zara_xgb,
        'LSTM (SFS+Google)': zara_lstm,
        'TFT (All Data)': zara_tft
    }
    
    # Train all Chanel models
    print("\n" + "="*70)
    print("TRAINING CHANEL MODELS")
    print("="*70)
    chanel_xgb = train_and_evaluate_xgboost(splits['chanel_xgb'], "Chanel XGBoost")
    chanel_lstm = train_and_evaluate_lstm(splits['chanel_lstm'], "Chanel LSTM", sequence_length=6, epochs=100)
    chanel_tft = train_and_evaluate_tft(splits['chanel_tft'], "Chanel TFT", sequence_length=6, epochs=100)
    
    chanel_results = {
        'XGBoost (SFS)': chanel_xgb,
        'LSTM (SFS+Google)': chanel_lstm,
        'TFT (All Data)': chanel_tft
    }
    
    # Compare
    zara_comparison = create_comparison_table(zara_results)
    chanel_comparison = create_comparison_table(chanel_results)
    
    print_analysis_summary(zara_comparison, chanel_comparison)
    create_summary_visualization(zara_results, chanel_results, 'plots/model_comparison.png')
    
    # Save results
    zara_comparison.to_csv('results/zara_comparison.csv')
    chanel_comparison.to_csv('results/chanel_comparison.csv')
    print("\nResults saved to results/")