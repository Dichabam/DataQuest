import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

def prepare_baseline_data(train_df, test_df, target_col='default_flag'):
    """Prepares raw data and applies standard scaling for the baseline model."""
    raw_features = ['age', 'annual_income', 'credit_utilisation_pct', 'dti_ratio', 
                    'loan_amount', 'home_ownership', 'loan_purpose', 'region']
    
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    X_train_raw = train_df[raw_features].copy()
    X_test_raw = test_df[raw_features].copy()
    
    for col in raw_features:
        if X_train_raw[col].dtype in ['float64', 'int64']:
            median_val = X_train_raw[col].median()
            X_train_raw[col] = X_train_raw[col].fillna(median_val)
            X_test_raw[col] = X_test_raw[col].fillna(median_val)
        else:
            X_train_raw[col] = X_train_raw[col].fillna('Missing')
            X_test_raw[col] = X_test_raw[col].fillna('Missing')
            
    num_cols = X_train_raw.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train_raw[num_cols] = scaler.fit_transform(X_train_raw[num_cols])
    X_test_raw[num_cols] = scaler.transform(X_test_raw[num_cols])
    
    X_train_encoded = pd.get_dummies(X_train_raw, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    return X_train_encoded, y_train, X_test_encoded, y_test


def prepare_woe_data(train_df, test_df, target_col='default_flag'):
    woe_features = [col for col in train_df.columns if col.endswith('_woe')]
    return train_df[woe_features], train_df[target_col], test_df[woe_features], test_df[target_col]


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name="Model", use_cv=False):
    """
    Trains the model. If use_cv=True, it performs 5-fold cross-validation 
    to find the mathematically optimal regularization strength (Hyperparameter Tuning).
    """
    if use_cv:
        # Cross-Validated Logistic Regression (Hyperparameter Tuning)
        model = LogisticRegressionCV(
            Cs=10, cv=5, penalty='l2', class_weight='balanced', 
            scoring='roc_auc', max_iter=1000, random_state=42
        )
    else:
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    gini_score = (auc_score * 2) - 1 
    
    print(f"--- {model_name} Performance ---")
    if use_cv:
        print(f"Optimal Regularization (C): {model.C_[0]:.4f}")
    print(f"ROC-AUC Score: {auc_score:.4f} | Gini: {gini_score:.4f}")
    
    return model, y_pred_proba


def apply_reject_inference(model, X_train, y_train, X_unlabeled):
    """
    Advanced Strategy: Reject Inference via Fuzzy Augmentation.
    Simulates predicting on rejected applicants and folding them back into training 
    to remove selection bias. (Used conceptually for the presentation).
    """
    # 1. Predict probabilities on previously rejected (unlabeled) applicants
    inferred_probs = model.predict_proba(X_unlabeled)[:, 1]
    
    # 2. Hard Cutoff approach: Assume highest 20% risk are definite defaults, lowest 20% are good.
    inferred_labels = (inferred_probs > 0.8).astype(int) 
    
    # 3. Retrain on the augmented dataset
    X_augmented = pd.concat([X_train, X_unlabeled])
    y_augmented = pd.concat([y_train, pd.Series(inferred_labels, index=X_unlabeled.index)])
    
    new_model = LogisticRegressionCV(cv=5, max_iter=1000, class_weight='balanced')
    new_model.fit(X_augmented, y_augmented)
    
    return new_model