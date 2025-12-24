"""
Main Runner Script: Fashion Trend Prediction Pipeline
======================================================

This script runs the complete ML pipeline for fashion trend prediction.

USAGE:
------
1. Place your data files in the same directory:
   - trend_counts_over_time.csv
   - google_trends.csv
   - California_weather.csv

2. Run this script:
   python main_runner.py

3. Results will be saved to:
   - plots/ directory (visualizations)
   - results/ directory (comparison tables)

PIPELINE STAGES:
----------------
1. Data Loading - Load and clean all datasets
2. EDA - Exploratory Data Analysis (optional, set RUN_EDA=True)
3. Feature Engineering - Create features for each model
4. Train/Test Split - Split data chronologically
5. Model Training - Train XGBoost, LSTM, TFT
6. Evaluation - Compare all models

WHAT EACH MODEL TESTS:
----------------------
- XGBoost (SFS only): Baseline - can we predict from history alone?
- LSTM (SFS + Google): Does search interest help predict trends?
- TFT (All data): Does weather/seasonality add further value?
"""

import os
import warnings
warnings.filterwarnings('ignore')

RUN_EDA = True  # Set to True to run exploratory data analysis
SEQUENCE_LENGTH = 6  # Lookback window for LSTM/TFT (in months)
TRAIN_FRACTION = 0.75  # Fraction of pre-peak data for training
EPOCHS = 100  # Training epochs for neural networks

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)


def main():
    """Run the complete pipeline."""
    
    print("="*70)
    print("FASHION TREND SATURATION PREDICTION")
    print("Comparing XGBoost, LSTM, and TFT")
    print("="*70)
    
    # ==================== STAGE 1: DATA LOADING ====================
    print("\n" + "="*70)
    print("STAGE 1: DATA LOADING")
    print("="*70)
    
    from load_data import load_all_data
    
    data = load_all_data(
        sfs_path='trend_counts_over_time.csv',
        google_path='google_trends.csv',
        weather_path='California_weather.csv'
    )
    
    # ==================== STAGE 2: EDA (Optional) ====================
    if RUN_EDA:
        print("\n" + "="*70)
        print("STAGE 2: EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        from eda import run_full_eda
        
        eda_results = run_full_eda(
            data['sfs'],
            data['google'],
            data['weather'],
            save_figures=True
        )
    
    # ==================== STAGE 3: FEATURE ENGINEERING ====================
    print("\n" + "="*70)
    print("STAGE 3: FEATURE ENGINEERING")
    print("="*70)
    
    from feature_eng import prepare_all_model_features
    
    feature_datasets = prepare_all_model_features(
        data['sfs'],
        data['google'],
        data['weather']
    )
    
    # ==================== STAGE 4: TRAIN/TEST SPLIT ====================
    print("\n" + "="*70)
    print("STAGE 4: TRAIN/TEST SPLIT")
    print("="*70)
    
    from train_test import prepare_all_splits
    
    splits = prepare_all_splits(
        feature_datasets,
        data['sfs'],
        train_fraction=TRAIN_FRACTION,
        visualize=True,
        save_plots=True
    )
    
    # ==================== STAGE 5: MODEL TRAINING ====================
    print("\n" + "="*70)
    print("STAGE 5: MODEL TRAINING")
    print("="*70)
    
    from xgboost import train_and_evaluate_xgboost, plot_predictions
    from lstm import train_and_evaluate_lstm, plot_lstm_predictions
    from tft import train_and_evaluate_tft, plot_tft_predictions
    
    all_results = {
        'zara': {},
        'chanel': {}
    }
    
    # ----- ZARA DRESS -----
    print("\n" + "-"*50)
    print("ZARA DRESS MODELS")
    print("-"*50)
    
    zara_xgb = train_and_evaluate_xgboost(
        splits['zara_xgb'],
        model_name="Zara XGBoost (SFS only)"
    )
    all_results['zara']['XGBoost (SFS only)'] = zara_xgb
    
    plot_predictions(
        zara_xgb,
        title='Zara Dress: XGBoost Predictions',
        save_path='plots/zara_xgb_predictions.png'
    )
    
    zara_lstm = train_and_evaluate_lstm(
        splits['zara_lstm'],
        model_name="Zara LSTM (SFS + Google)",
        sequence_length=SEQUENCE_LENGTH,
        epochs=EPOCHS
    )
    all_results['zara']['LSTM (SFS + Google)'] = zara_lstm
    
    plot_lstm_predictions(
        zara_lstm,
        splits['zara_lstm']['y_train'],
        splits['zara_lstm']['train_dates'],
        title='Zara Dress: LSTM Predictions',
        save_path='plots/zara_lstm_predictions.png'
    )
    
    zara_tft = train_and_evaluate_tft(
        splits['zara_tft'],
        model_name="Zara TFT (All Data)",
        sequence_length=SEQUENCE_LENGTH,
        epochs=EPOCHS
    )
    all_results['zara']['TFT (All Data)'] = zara_tft
    
    plot_tft_predictions(
        zara_tft,
        splits['zara_tft']['y_train'],
        splits['zara_tft']['train_dates'],
        title='Zara Dress: TFT Predictions',
        save_path='plots/zara_tft_predictions.png'
    )
    
    # ----- CHANEL BAG -----
    print("\n" + "-"*50)
    print("CHANEL BAG MODELS")
    print("-"*50)

    chanel_xgb = train_and_evaluate_xgboost(
        splits['chanel_xgb'],
        model_name="Chanel XGBoost (SFS only)"
    )
    all_results['chanel']['XGBoost (SFS only)'] = chanel_xgb
    
    plot_predictions(
        chanel_xgb,
        title='Chanel Bag: XGBoost Predictions',
        save_path='plots/chanel_xgb_predictions.png'
    )
    
    chanel_lstm = train_and_evaluate_lstm(
        splits['chanel_lstm'],
        model_name="Chanel LSTM (SFS + Google)",
        sequence_length=SEQUENCE_LENGTH,
        epochs=EPOCHS
    )
    all_results['chanel']['LSTM (SFS + Google)'] = chanel_lstm
    
    plot_lstm_predictions(
        chanel_lstm,
        splits['chanel_lstm']['y_train'],
        splits['chanel_lstm']['train_dates'],
        title='Chanel Bag: LSTM Predictions',
        save_path='plots/chanel_lstm_predictions.png'
    )
    
    chanel_tft = train_and_evaluate_tft(
        splits['chanel_tft'],
        model_name="Chanel TFT (All Data)",
        sequence_length=SEQUENCE_LENGTH,
        epochs=EPOCHS
    )
    all_results['chanel']['TFT (All Data)'] = chanel_tft
    
    plot_tft_predictions(
        chanel_tft,
        splits['chanel_tft']['y_train'],
        splits['chanel_tft']['train_dates'],
        title='Chanel Bag: TFT Predictions',
        save_path='plots/chanel_tft_predictions.png'
    )
    
    # ==================== STAGE 6: EVALUATION ====================
    print("\n" + "="*70)
    print("STAGE 6: MODEL COMPARISON")
    print("="*70)
    
    from evaluation import (
        create_comparison_table,
        create_summary_visualization,
        print_analysis_summary
    )

    zara_comparison = create_comparison_table(all_results['zara'])
    chanel_comparison = create_comparison_table(all_results['chanel'])
  
    print_analysis_summary(zara_comparison, chanel_comparison)
    
    create_summary_visualization(
        all_results['zara'],
        all_results['chanel'],
        save_path='plots/model_comparison_summary.png'
    )
    
    zara_comparison.to_csv('results/zara_model_comparison.csv')
    chanel_comparison.to_csv('results/chanel_model_comparison.csv')
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    print("\nFILES CREATED:")
    print("\nPlots (in plots/ directory):")
    print("  - trend_lifecycles.png")
    print("  - sfs_vs_google.png")
    print("  - weather_patterns.png")
    print("  - weather_seasonality.png")
    print("  - zara_split.png, chanel_split.png")
    print("  - zara_xgb_predictions.png, chanel_xgb_predictions.png")
    print("  - zara_lstm_predictions.png, chanel_lstm_predictions.png")
    print("  - zara_tft_predictions.png, chanel_tft_predictions.png")
    print("  - model_comparison_summary.png")
    
    print("\nResults (in results/ directory):")
    print("  - zara_model_comparison.csv")
    print("  - chanel_model_comparison.csv")
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR YOUR RESEARCH")
    print("="*70)
    print("""
    1. Review the comparison tables to determine:
       - Which model performs best for each trend
       - Whether external data (Google, Weather) improves predictions
    
    2. Analyze feature importance:
       - Run XGBoost feature importance to see which lags matter most
       - Run TFT feature importance to see which data sources matter
    
    3. Consider hyperparameter tuning:
       - Adjust sequence_length (try 3, 6, 12 months)
       - Adjust model architectures
       - Try different train/test split ratios
    
    4. Write up your findings:
       - Compare models quantitatively (RMSE, R2)
       - Discuss which data sources add value
       - Analyze why certain models work better for certain trends
    """)
    
    return all_results, zara_comparison, chanel_comparison


if __name__ == "__main__":
    results, zara_comp, chanel_comp = main()