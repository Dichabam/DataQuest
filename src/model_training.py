import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def prepare_baseline_data(train_df, test_df, target_col='default_flag'):
    """
    Prepares raw data for the baseline model. 
    Logistic regression cannot handle text (categories) or missing values natively.
    """
    raw_features = ['age', 'annual_income', 'credit_utilisation_pct', 'dti_ratio', 
                    'loan_amount', 'home_ownership', 'loan_purpose', 'region']
    
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # Use .copy() to avoid SettingWithCopy warnings
    X_train_raw = train_df[raw_features].copy()
    X_test_raw = test_df[raw_features].copy()
    
    # --- THE FIX: Handle missing values for the strict Baseline Model ---
    for col in raw_features:
        if X_train_raw[col].dtype in ['float64', 'int64']:
            # Fill missing numbers with the median
            median_val = X_train_raw[col].median()
            X_train_raw[col] = X_train_raw[col].fillna(median_val)
            X_test_raw[col] = X_test_raw[col].fillna(median_val)
        else:
            # Fill missing text with 'Missing'
            X_train_raw[col] = X_train_raw[col].fillna('Missing')
            X_test_raw[col] = X_test_raw[col].fillna('Missing')
    # --------------------------------------------------------------------
    
    # One-Hot Encode categorical variables
    X_train_encoded = pd.get_dummies(X_train_raw, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True)
    
    # Align columns just in case the test set is missing a category present in train
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    return X_train_encoded, y_train, X_test_encoded, y_test

def prepare_woe_data(train_df, test_df, target_col='default_flag'):
    """
    Prepares the engineered data for the advanced model.
    Grabs only the columns ending in '_woe' that we created in Phase 2.
    """
    woe_features = [col for col in train_df.columns if col.endswith('_woe')]
    
    X_train_woe = train_df[woe_features]
    y_train = train_df[target_col]
    
    X_test_woe = test_df[woe_features]
    y_test = test_df[target_col]
    
    return X_train_woe, y_train, X_test_woe, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains a Logistic Regression model and prints standard credit risk evaluation metrics.
    """
    # Initialize the model with mild L2 regularization (Ridge) to prevent overfitting
    # class_weight='balanced' helps if we have far more "good" loans than "bad" loans
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    # Train the manager!
    model.fit(X_train, y_train)
    
    # Predict probabilities (e.g., 85% chance of default)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Predict hard classes (0 or 1) based on a default 50% threshold
    y_pred_class = model.predict(X_test)
    
    # Calculate ROC-AUC (The industry standard for credit models)
    # AUC ranges from 0.5 (random guessing) to 1.0 (perfect prediction)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    gini_score = (auc_score * 2) - 1 # Gini is commonly used in banking
    
    print(f"--- {model_name} Performance ---")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"Gini Coefficient: {gini_score:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_class))
    
    return model, y_pred_proba

if __name__ == "__main__":
    # Note: In a real run, train_df and test_df would be passed from feature_engineering.py
    # X_train_base, y_train, X_test_base, y_test = prepare_baseline_data(train_df, test_df)
    # baseline_model, base_preds = train_and_evaluate(X_train_base, y_train, X_test_base, y_test, "Baseline Model")
    
    # X_train_woe, y_train, X_test_woe, y_test = prepare_woe_data(train_df, test_df)
    # advanced_model, woe_preds = train_and_evaluate(X_train_woe, y_train, X_test_woe, y_test, "Advanced WoE Model")
    pass