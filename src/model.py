# Baseline model 

from xgboost import XGBClassifier


def build_xgb_model(params=None):

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }

    if params is not None:
        default_params.update(params)

    model = XGBClassifier(**default_params)

    return model
