import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import concurrent.futures

def evaluate_model(params, train_x, train_y, skf):
    """Evaluate a model with the given parameters using cross-validation."""
    train_set = lgb.Dataset(train_x, label=train_y)
    
    # Perform cross-validation
    cv_results = lgb.cv(
        params,
        train_set,
        num_boost_round=2000,
        folds=skf.split(train_x, train_y),
        metrics='auc',
        seed=42
    )
    
    # Get the best AUC score
    score = max(cv_results['valid auc-mean'])
    return score

def main():
    # Load dataset
    df = pd.read_csv('/home/jl2815/ds_projects/travelers/trav_dataset1.csv')
    # test_set = df.sample(frac=0.2, random_state=24)
    test_set = df[df['split']=='Test']
    test_mask = df.index.isin(test_set.index)
    train_set = df.loc[~test_mask,:].reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    train_set = train_set.drop(columns=['split'])
    test_set = test_set.drop(columns=['split'])

    # Separate features and target from the entire training set
    train_y = train_set['convert_ind']
    train_x = train_set.drop(columns=['convert_ind'])

    test_y = test_set['convert_ind']
    test_x = test_set.drop(columns=['convert_ind'])

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'num_leaves': np.arange(10, 20),
        'learning_rate': [0.005, 0.01, 0.02, 0.03],
        'feature_fraction': [0.75, 0.8, 0.85],
        'min_data_in_leaf': [10, 20, 30],
        'lambda_l1': [0, 0.1, 1],
        'lambda_l2': [0, 0.1, 1],
        'early_stopping_rounds':50
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_score = 0
    best_params = None

    # Create a list of all parameter combinations
    param_combinations = [
        {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'auc',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'min_data_in_leaf': min_data_in_leaf,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'is_unbalance': True
        }
        for num_leaves in param_grid['num_leaves']
        for learning_rate in param_grid['learning_rate']
        for feature_fraction in param_grid['feature_fraction']
        for min_data_in_leaf in param_grid['min_data_in_leaf']
        for lambda_l1 in param_grid['lambda_l1']
        for lambda_l2 in param_grid['lambda_l2']
    ]

    # Evaluate parameter combinations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(evaluate_model, params, train_x, train_y, skf)
            for params in param_combinations
        ]
        for future, params in zip(concurrent.futures.as_completed(futures), param_combinations):
            score = future.result()
            print(f"Params: {params}, AUC: {score}")
            if score > best_score:
                best_score = score
                best_params = params

    # Print the best parameters and score
    print("Best Parameters:", best_params)
    print("Best CV AUC Score:", best_score)

if __name__ == '__main__':
    main()
