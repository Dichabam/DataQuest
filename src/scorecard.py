import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    """
    Extracts the weights (coefficients) from the Logistic Regression model.
    Because our features are WoE-transformed (meaning they are on the same scale),
    we can directly compare the magnitude of the coefficients.
    """
    # Extract coefficients
    coefficients = model.coef_[0]
    
    # Create a DataFrame for readability
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Importance': np.abs(coefficients)
    })
    
    # Sort by most important
    importance_df = importance_df.sort_values(by='Absolute_Importance', ascending=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Coefficient'], color='skyblue')
    plt.xlabel('Coefficient Value (Impact on Default Risk)')
    plt.title('Feature Importance: What drives the model?')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.tight_layout()
    # plt.show() # Uncomment to view when running
    
    return importance_df.sort_values(by='Absolute_Importance', ascending=False)

def generate_scorecard(model, woe_engine, feature_names, target_score=600, target_odds=50, pdo=20):
    """
    Translates the model's coefficients into a traditional points-based Scorecard.
    
    Parameters:
    - target_score: Baseline score (e.g., 600 points)
    - target_odds: The base odds of "Good" to "Bad" at the target score (e.g., 50 to 1)
    - pdo (Points to Double Odds): How many points are needed to halve the risk (e.g., 20 points)
    """
    # 1. Calculate scaling factors
    factor = pdo / np.log(2)
    offset = target_score - (factor * np.log(target_odds))
    
    # 2. Extract model intercept and coefficients
    intercept = model.intercept_[0]
    coefficients = dict(zip(feature_names, model.coef_[0]))
    n_features = len(feature_names)
    
    scorecard = []
    
    # 3. Calculate points for each WoE bin across all features
    for feature in feature_names:
        original_feature = feature.replace('_woe', '')
        
        # Get the WoE mappings we saved in Phase 2
        woe_dict = woe_engine.woe_dicts[original_feature]
        model_coef = coefficients[feature]
        
        for bin_name, woe_value in woe_dict.items():
            # The scorecard mathematical formula:
            # Points = -(WoE * Coefficient * Factor) + (Offset / n_features)
            
            # Note: We subtract because a higher score should mean LOWER risk, 
            # but our model predicts the probability of DEFAULT (higher = bad).
            points = -(woe_value * model_coef * factor) + (offset / n_features)
            
            scorecard.append({
                'Feature': original_feature,
                'Bin': bin_name,
                'WoE': round(woe_value, 4),
                'Points': round(points)
            })
            
    # Combine into a clean DataFrame
    scorecard_df = pd.DataFrame(scorecard)
    
    return scorecard_df

if __name__ == "__main__":
    # Note: In a real script, 'advanced_model', 'woe_engine', and 'X_train_woe' 
    # would be imported or passed from the previous phases.
    
    # importance = plot_feature_importance(advanced_model, X_train_woe.columns)
    # print("Top Features:\n", importance.head())
    
    # my_scorecard = generate_scorecard(advanced_model, woe_engine, X_train_woe.columns)
    # print("\nSample Scorecard Rules:\n", my_scorecard.head(10))
    pass