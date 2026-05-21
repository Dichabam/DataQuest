# test_pipeline.py
"""
Integration tests for the credit risk pipeline.

FIX: Previously this file had no assert statements — it was a script that
printed output but would not catch regressions (a passing run looked identical
to a broken one as long as no exception was thrown). Each phase now has
explicit assertions on shapes, types, value ranges, and invariants.

Run with:
    python test_pipeline.py
or (if you add pytest):
    pytest test_pipeline.py -v
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.data_prep import load_and_split_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_baseline_data, prepare_woe_data, train_and_evaluate
from src.scorecard import plot_feature_importance, generate_scorecard
from src.business_logic import calculate_portfolio_profit, audit_fairness


def _assert(condition: bool, message: str):
    """Thin wrapper so failures are readable without pytest."""
    if not condition:
        raise AssertionError(f"FAIL: {message}")
    print(f"  ✓ {message}")


def test_data_prep():
    print("\n[1/4] Testing Data Preparation...")

    train, test = load_and_split_data("data/raw/loan_book.csv")

    _assert(len(train) > 0, "Train set is non-empty")
    _assert(len(test) > 0, "Test set is non-empty")
    _assert('set' not in train.columns, "'set' column dropped from train")
    _assert('set' not in test.columns, "'set' column dropped from test")

    train_clean, imputers = clean_data(train, is_train=True)
    test_clean, _ = clean_data(test, is_train=False, imputers=imputers)

    # Imputer keys are all present
    required_keys = {'months_since_last_delinquency_fill', 'pct_accounts_current_median', 'income_cap'}
    _assert(required_keys.issubset(imputers.keys()), "All imputer keys present")

    # Fill value is applied — no nulls in these columns on either split
    _assert(
        train_clean['months_since_last_delinquency'].isna().sum() == 0,
        "No nulls in months_since_last_delinquency (train)",
    )
    _assert(
        test_clean['months_since_last_delinquency'].isna().sum() == 0,
        "No nulls in months_since_last_delinquency (test)",
    )
    _assert(
        train_clean['pct_accounts_current'].isna().sum() == 0,
        "No nulls in pct_accounts_current (train)",
    )

    # Income cap applied: no value should exceed the 99th percentile computed on train
    _assert(
        train_clean['annual_income'].max() <= imputers['income_cap'] + 1e-6,
        "Income cap applied correctly on train",
    )
    _assert(
        test_clean['annual_income'].max() <= imputers['income_cap'] + 1e-6,
        "Income cap applied correctly on test (uses train cap, not test cap)",
    )

    # home_ownership standardised to uppercase
    _assert(
        train_clean['home_ownership'].str.isupper().all(),
        "home_ownership is uppercase on train",
    )

    # Passing imputers to a train=True call should raise
    try:
        clean_data(train, is_train=True, imputers=imputers)
        _assert(False, "Should have raised ValueError for imputers on train call")
    except ValueError:
        _assert(True, "ValueError raised when passing imputers to is_train=True")

    return train_clean, test_clean


def test_feature_engineering(train_clean, test_clean):
    print("\n[2/4] Testing Feature Engineering...")

    train_woe, test_woe, woe_engine = engineer_features(train_clean, test_clean)

    woe_cols = [c for c in train_woe.columns if c.endswith('_woe')]
    _assert(len(woe_cols) > 0, "WoE columns were created")

    # No NaNs in WoE columns on train
    for col in woe_cols:
        _assert(
            train_woe[col].isna().sum() == 0,
            f"No NaNs in {col} (train)",
        )

    # Test WoE columns should also have no NaNs (unseen bins filled with 0)
    for col in woe_cols:
        _assert(
            test_woe[col].isna().sum() == 0,
            f"No NaNs in {col} (test)",
        )

    # IV scores exist for each feature
    expected_features = ['age', 'annual_income', 'credit_utilisation_pct', 'dti_ratio',
                         'loan_amount', 'home_ownership', 'loan_purpose', 'region']
    for feat in expected_features:
        _assert(feat in woe_engine.iv_scores, f"IV score exists for '{feat}'")
        _assert(woe_engine.iv_scores[feat] >= 0, f"IV for '{feat}' is non-negative")

    return train_woe, test_woe, woe_engine


def test_model_training(train_clean, test_clean, train_woe, test_woe):
    print("\n[3/4] Testing Model Training...")

    # Baseline model
    X_train_base, y_train, X_test_base, y_test = prepare_baseline_data(train_clean, test_clean)
    base_model, base_probs = train_and_evaluate(
        X_train_base, y_train, X_test_base, y_test, "Baseline Model"
    )

    _assert(len(base_probs) == len(y_test), "Baseline: prediction length matches test set")
    _assert(base_probs.min() >= 0.0 and base_probs.max() <= 1.0,
            "Baseline: probabilities in [0, 1]")

    # WoE model
    X_train_woe, y_train_woe, X_test_woe, y_test_woe = prepare_woe_data(train_woe, test_woe)
    adv_model, adv_probs = train_and_evaluate(
        X_train_woe, y_train_woe, X_test_woe, y_test_woe, "Advanced WoE Model"
    )

    _assert(len(adv_probs) == len(y_test_woe), "WoE model: prediction length matches test set")
    _assert(adv_probs.min() >= 0.0 and adv_probs.max() <= 1.0,
            "WoE model: probabilities in [0, 1]")

    return adv_model, X_train_woe, X_test_woe, y_test_woe, adv_probs


def test_scorecard(adv_model, woe_engine, X_train_woe):
    print("\n[4/4] Testing Scorecard Generation...")

    importance_df = plot_feature_importance(adv_model, X_train_woe.columns)
    _assert('Coefficient' in importance_df.columns, "Importance df has Coefficient column")
    _assert(len(importance_df) == len(X_train_woe.columns), "One row per feature")

    scorecard_df = generate_scorecard(adv_model, woe_engine, X_train_woe.columns)
    _assert('Feature' in scorecard_df.columns, "Scorecard has Feature column")
    _assert('Points' in scorecard_df.columns, "Scorecard has Points column")
    _assert('WoE' in scorecard_df.columns, "Scorecard has WoE column")
    _assert(len(scorecard_df) > 0, "Scorecard is non-empty")

    # Points should be finite numbers
    _assert(
        np.isfinite(scorecard_df['Points'].values).all(),
        "All scorecard points are finite",
    )

    return scorecard_df


def test_business_logic(y_test, adv_probs, test_clean, scorecard_df):
    print("\n  Testing Business Logic...")

    # Profit calculation
    net_profit, rev, losses, app_rate = calculate_portfolio_profit(
        y_true=y_test,
        y_prob=pd.Series(adv_probs),
        cutoff_prob=0.4,
        loan_amounts=test_clean['loan_amount'],
    )

    _assert(isinstance(net_profit, float), "Net profit is a float")
    _assert(rev >= 0, "Revenue is non-negative")
    _assert(losses >= 0, "Losses are non-negative")
    _assert(0 <= app_rate <= 100, "Approval rate is a percentage in [0, 100]")

    # Fairness audit
    test_clean = test_clean.copy()
    test_clean['credit_score'] = 600  # dummy score for audit test
    fairness_df = audit_fairness(test_clean, 'region', 'credit_score', cutoff_score=600)

    _assert('Approval_Rate' in fairness_df.columns, "Fairness df has Approval_Rate")
    _assert(
        (fairness_df['Approval_Rate'] >= 0).all() and (fairness_df['Approval_Rate'] <= 100).all(),
        "All approval rates in [0, 100]",
    )


def run_tests():
    print("=" * 50)
    print("  CREDIT RISK PIPELINE — INTEGRATION TESTS")
    print("=" * 50)

    train_clean, test_clean = test_data_prep()
    train_woe, test_woe, woe_engine = test_feature_engineering(train_clean, test_clean)
    adv_model, X_train_woe, X_test_woe, y_test, adv_probs = test_model_training(
        train_clean, test_clean, train_woe, test_woe
    )
    scorecard_df = test_scorecard(adv_model, woe_engine, X_train_woe)
    test_business_logic(y_test, adv_probs, test_clean, scorecard_df)

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()