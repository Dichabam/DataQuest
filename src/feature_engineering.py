import pandas as pd
import numpy as np

class WoE_Binning:
    def __init__(self, target_col='default_flag'):
        self.target_col = target_col
        self.woe_dicts = {}
        self.iv_scores = {}
        self.bins = {}

    def _calculate_woe_iv(self, df, feature, target):
        """Calculates WoE and IV for a single categorical/binned feature."""
        grouped = df.groupby(feature)[target].agg(['count', 'sum'])
        grouped.columns = ['Total', 'Bad']
        grouped['Good'] = grouped['Total'] - grouped['Bad']
        
        total_bad = grouped['Bad'].sum()
        total_good = grouped['Good'].sum()
        
        # Calculate distributions (add small epsilon to avoid division by zero)
        grouped['Dist_Good'] = np.maximum(grouped['Good'], 0.5) / total_good
        grouped['Dist_Bad'] = np.maximum(grouped['Bad'], 0.5) / total_bad
        
        # Calculate WoE and IV
        grouped['WoE'] = np.log(grouped['Dist_Good'] / grouped['Dist_Bad'])
        grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']
        
        return grouped['WoE'].to_dict(), grouped['IV'].sum()

    def fit_transform_continuous(self, df, feature, q=5):
        """Bins a continuous feature and calculates WoE."""
        # Create quantile bins
        df[f'{feature}_binned'], bins = pd.qcut(df[feature], q=q, retbins=True, duplicates='drop')
        
        # Format bin labels to strings for mapping
        df[f'{feature}_binned'] = df[f'{feature}_binned'].astype(str)
        self.bins[feature] = bins
        
        woe_dict, iv = self._calculate_woe_iv(df, f'{feature}_binned', self.target_col)
        self.woe_dicts[feature] = woe_dict
        self.iv_scores[feature] = iv
        
        # Map WoE values
        df[f'{feature}_woe'] = df[f'{feature}_binned'].map(woe_dict)
        return df

    def fit_transform_categorical(self, df, feature):
        """Calculates WoE for a categorical feature."""
        woe_dict, iv = self._calculate_woe_iv(df, feature, self.target_col)
        self.woe_dicts[feature] = woe_dict
        self.iv_scores[feature] = iv
        
        df[f'{feature}_woe'] = df[feature].map(woe_dict)
        return df

    def transform(self, df, feature, is_continuous=False):
        """Applies learned WoE mappings to a new (test) dataset."""
        if is_continuous:
            # Apply saved bins to test data
            df[f'{feature}_binned'] = pd.cut(df[feature], bins=self.bins[feature], include_lowest=True).astype(str)
            df[f'{feature}_woe'] = df[f'{feature}_binned'].map(self.woe_dicts[feature])
            # Handle out of bounds / unseen bins by filling with 0 (neutral risk)
            df[f'{feature}_woe'] = df[f'{feature}_woe'].fillna(0)
        else:
            df[f'{feature}_woe'] = df[feature].map(self.woe_dicts[feature]).fillna(0)
        return df

def engineer_features(train_df, test_df):
    """Orchestrates the feature engineering process."""
    woe_engine = WoE_Binning(target_col='default_flag')
    
    continuous_features = ['age', 'annual_income', 'credit_utilisation_pct', 'dti_ratio', 'loan_amount']
    categorical_features = ['home_ownership', 'loan_purpose', 'region']
    
    print("Processing Training Data...")
    for col in continuous_features:
        train_df = woe_engine.fit_transform_continuous(train_df, col, q=5)
        
    for col in categorical_features:
        train_df = woe_engine.fit_transform_categorical(train_df, col)
        
    print("Applying mappings to Test Data...")
    for col in continuous_features:
        test_df = woe_engine.transform(test_df, col, is_continuous=True)
        
    for col in categorical_features:
        test_df = woe_engine.transform(test_df, col, is_continuous=False)
        
    # Print Information Values to show which features are most predictive
    print("\nFeature Information Values (IV):")
    for feat, iv in sorted(woe_engine.iv_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{feat}: {iv:.4f} " + ("(Strong)" if iv > 0.3 else "(Medium)" if iv > 0.1 else "(Weak)"))
        
    return train_df, test_df, woe_engine