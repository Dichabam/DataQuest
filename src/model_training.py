import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

def prepare_baseline_data(train_df, test_df, target_col='default_flag'):
    raw_features = ['age', 'annual_income', 'credit_utilisation_pct', 'dti_ratio', 
                    'loan_amount', 'home_ownership', 'loan_purpose', 'region',
                    'months_since_last_delinquency', 'pct_accounts_current']
    
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
    if use_cv:
        model = LogisticRegressionCV(
            Cs=10, cv=5, penalty='l2', class_weight='balanced', 
            scoring='roc_auc', max_iter=1000, random_state=42
        )
    else:
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    return model, y_pred_proba