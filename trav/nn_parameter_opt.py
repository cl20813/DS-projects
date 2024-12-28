import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
from itertools import product
import concurrent.futures

def evaluate_model(params, X_train, y_train, X_test, y_test):
    """
    Function to evaluate a single combination of TabNet parameters.
    """
    n_d, n_a, n_steps, gamma, lambda_sparse, lr = params

    # Define the model
    model = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_params=dict(lr=lr),
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc'],
        max_epochs=100,
        batch_size=128,
        patience=10
    )

    # Evaluate the model
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    return auc


def main():
    # Load dataset
    df = pd.read_csv('/home/jl2815/ds_projects/travelers/trav_dataset1.csv')

    # Split the data
    train_set, test_set = train_test_split(df, test_size=0.2, stratify=df['convert_ind'], random_state=24)

    # Drop the 'split' column if it exists
    if 'split' in train_set.columns:
        train_set = train_set.drop(columns=['split'])
        test_set = test_set.drop(columns=['split'])

    # Separate features and target
    y_train = train_set['convert_ind'].values
    train_x = train_set.drop(columns=['convert_ind'])

    y_test = test_set['convert_ind'].values
    test_x = test_set.drop(columns=['convert_ind'])

    # Select numeric columns
    train_x = train_x.select_dtypes(include=[np.number])
    test_x = test_x.select_dtypes(include=[np.number])

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_x)
    X_test = scaler.transform(test_x)

    # Ensure inputs are NumPy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Define the parameter grid
    param_grid = {
        'n_d': [8, 16, 32],
        'n_a': [8, 16, 32],
        'n_steps': [3, 5, 7],
        'gamma': [1.0, 1.5, 2.0],
        'lambda_sparse': [1e-4, 1e-3, 1e-2],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    # Create all combinations of parameters
    param_combinations = list(product(
        param_grid['n_d'],
        param_grid['n_a'],
        param_grid['n_steps'],
        param_grid['gamma'],
        param_grid['lambda_sparse'],
        param_grid['learning_rate']
    ))

    # Initialize variables to track the best parameters
    best_auc = 0
    best_params = None

    # Evaluate parameter combinations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(evaluate_model, params, X_train, y_train, X_test, y_test)
            for params in param_combinations
        ]
        for future, params in zip(concurrent.futures.as_completed(futures), param_combinations):
            try:
                score = future.result()
                print(f"Params: {params}, AUC: {score:.4f}")
                if score > best_auc:
                    best_auc = score
                    best_params = params
            except Exception as e:
                print(f"Error evaluating params {params}: {e}")

    # Print the best parameters and AUC
    print("Best AUC:", best_auc)
    print("Best Parameters:", best_params)


if __name__ == '__main__':
    main()

