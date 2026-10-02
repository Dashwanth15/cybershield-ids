import os
import time
import json
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR

def train_baseline_model(X_train, y_train):
    """
    Trains a baseline XGBoost model with standard default parameters.
    Records exact training time.
    """
    print("\n--- Training Baseline XGBoost Model ---")
    start_time = time.time()
    
    baseline_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    baseline_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Baseline XGBoost training completed in {train_time:.2f} seconds.")
    
    # Save baseline model
    os.makedirs(MODELS_DIR, exist_ok=True)
    baseline_path = os.path.join(MODELS_DIR, 'baseline_xgb.json')
    baseline_model.save_model(baseline_path)
    print(f"Baseline model saved to {baseline_path}")
    
    return baseline_model, train_time

def tune_xgboost_model(X_train, y_train, cv=5, n_iter=15):
    """
    Performs hyperparameter optimization using RandomizedSearchCV with StratifiedKFold.
    Optimizes for F1-score to balance attack detection (recall) with false alarm rate (precision).
    """
    print(f"\n--- Hyperparameter Tuning (RandomizedSearchCV, {n_iter} iterations, {cv}-fold CV) ---")
    
    param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.0, 0.1, 0.2],
        'reg_alpha': [0.0, 0.1, 1.0],
        'reg_lambda': [1.0, 2.0, 5.0],
        'scale_pos_weight': [1.0, 1.15]
    }
    
    base_xgb = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv_strategy,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1 = float(search.best_score_)
    
    print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds.")
    print(f"Best CV F1-Score: {best_cv_f1:.4f}")
    print(f"Best Hyperparameters:\n{json.dumps(best_params, indent=2)}")
    
    # Save tuned model
    os.makedirs(MODELS_DIR, exist_ok=True)
    tuned_path = os.path.join(MODELS_DIR, 'tuned_xgb.json')
    best_model.save_model(tuned_path)
    print(f"Tuned model saved to {tuned_path}")
    
    # Save tuning summary
    os.makedirs(REPORTS_DIR, exist_ok=True)
    tuning_info = {
        'best_hyperparameters': best_params,
        'best_cv_f1': best_cv_f1,
        'tuning_time_seconds': tuning_time,
        'n_iterations': n_iter,
        'cv_folds': cv
    }
    with open(os.path.join(REPORTS_DIR, 'tuning_summary.json'), 'w') as f:
        json.dump(tuning_info, f, indent=2)
        
    return best_model, best_params, tuning_time
