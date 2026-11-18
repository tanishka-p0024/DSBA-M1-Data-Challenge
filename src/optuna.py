
import optuna
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score


def objective(trial, X, y, groups):

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,

        "n_estimators": trial.suggest_int(
            "n_estimators", 200, 2000
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 7
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),

        "gamma": trial.suggest_float(
            "gamma", 0, 5
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10, log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-4, 10, log=True
        )
    }

    model = XGBClassifier(**params)

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="f1",
        n_jobs=1
    )

    return scores.mean()


def tune_xgb(X, y, groups, n_trials=50):

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, groups),
        n_trials=n_trials
    )

    return study

