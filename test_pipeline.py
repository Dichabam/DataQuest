# test_pipeline.py
import pandas as pd
import warnings

# Suppress warnings for cleaner output during testing
warnings.filterwarnings('ignore')

# Import our custom modules from the src folder
from src.data_prep import load_and_split_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_baseline_data, prepare_woe_data, train_and_evaluate
from src.scorecard import plot_feature_importance, generate_scorecard

def run_tests():
    print("--- STARTING PIPELINE TEST ---")
    
    # ---------------------------------------------------------
    # PHASE 1: DATA PREPARATION
    # ---------------------------------------------------------
    print("\n[1/4] Loading and Cleaning Data...")
    # Make sure this path points to where you saved your loan_book.csv
    train, test = load_and_split_data("data/raw/loan_book.csv")
    train_clean, imputers = clean_data(train, is_train=True)
    test_clean, _ = clean_data(test, is_train=False, imputers=imputers)
    print(f"✓ Data loaded. Training rows: {len(train_clean)}, Test rows: {len(test_clean)}")

    # ---------------------------------------------------------
    # PHASE 2: ADVANCED FEATURE ENGINEERING (WoE)
    # ---------------------------------------------------------
    print("\n[2/4] Engineering WoE Features...")
    train_woe, test_woe, woe_engine = engineer_features(train_clean, test_clean)
    print("✓ WoE features successfully generated.")

    # ---------------------------------------------------------
    # PHASE 3: MODEL TRAINING & EVALUATION
    # ---------------------------------------------------------
    print("\n[3/4] Training Logistic Regression Models...")
    
    # 1. Baseline Model
    X_train_base, y_train, X_test_base, y_test = prepare_baseline_data(train_clean, test_clean)
    print("\nTraining Baseline (Raw Data) Model:")
    base_model, _ = train_and_evaluate(X_train_base, y_train, X_test_base, y_test, "Baseline Model")
    
    # 2. Advanced WoE Model
    X_train_woe, y_train, X_test_woe, y_test = prepare_woe_data(train_woe, test_woe)
    print("\nTraining Advanced (WoE) Model:")
    adv_model, _ = train_and_evaluate(X_train_woe, y_train, X_test_woe, y_test, "Advanced WoE Model")

    # ---------------------------------------------------------
    # PHASE 4: EXPLAINABILITY & SCORECARD
    # ---------------------------------------------------------
    print("\n[4/4] Generating Scorecard and Feature Importance...")
    
    # Feature Importance
    importance_df = plot_feature_importance(adv_model, X_train_woe.columns)
    print("\nTop 3 Most Important Features:")
    print(importance_df.head(3).to_string(index=False))
    
    # Generate Scorecard
    scorecard_df = generate_scorecard(adv_model, woe_engine, X_train_woe.columns)
    print("\nSample Scorecard Rules (First 10 rows):")
    print(scorecard_df.head(10).to_string(index=False))
    
    print("\n--- PIPELINE TEST COMPLETE ---")
    print("If you see a bar chart pop up, feature importance plotted successfully!")
    import matplotlib.pyplot as plt
    plt.show() # This will display the feature importance chart

if __name__ == "__main__":
    run_tests()