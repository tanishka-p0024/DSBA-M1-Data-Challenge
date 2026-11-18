
import optuna
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score


def objective(trial, X, y, groups, cat_features):

    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,

        "iterations": trial.suggest_int(
            "iterations", 500, 1500
        ),

        "depth": trial.suggest_int(
            "depth", 4, 10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 1, log=True
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1, 10, log=True
        ),

        "random_strength": trial.suggest_float(
            "random_strength", 0, 2
        ),

        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0, 5
        ),

        "border_count": trial.suggest_int(
            "border_count", 32, 255
        )
    }

    model = CatBoostClassifier(
        **params,
        cat_features=cat_features
    )

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = []

    for train_idx, val_idx in cv.split(X, y, groups):

        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_val = y.iloc[val_idx]

        model_fold = CatBoostClassifier(
            **params,
            cat_features=cat_features
        )

        model_fold.fit(
            X_fold_train,
            y_fold_train
        )

        probabilities = model_fold.predict_proba(
            X_fold_val
        )[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        score = f1_score(y_fold_val, predictions)

        scores.append(score)

    return np.mean(scores)


def tune_catboost(X, y, groups, cat_features, n_trials=50):

    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
            groups,
            cat_features
        ),
        n_trials=n_trials,show_progress_bar=True
    )

    return study
