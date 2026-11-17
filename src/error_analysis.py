"""Simple error analysis for the DSBA-M1 concert-ticket baseline models.

This module only evaluates predictions. It never trains or tunes a model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def analyze_models(y_true, predictions, features, output_dir="evaluation_results"):
    """Compare model predictions and examine errors by user characteristics.

    Parameters
    ----------
    y_true : array-like
        True 0/1 labels from the validation set (y_val).
    predictions : dict
        Model name -> predicted 0/1 labels, in the same row order as y_true.
    features : pandas.DataFrame
        Original validation features (X_val), before encoding or scaling.
    output_dir : str
        Folder for the exported tables and figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = np.asarray(y_true).reshape(-1)
    if not set(np.unique(actual)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1.")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be the original X_val pandas DataFrame.")
    if len(features) != len(actual):
        raise ValueError("X_val and y_val must have the same number of rows.")
    if not predictions:
        raise ValueError("Please provide at least one model's predictions.")

    summary_rows = []
    group_tables = []
    # Keep only a few understandable user characteristics to start with.
    group_columns = ["first_timer", "wishlisted", "has_music_app", "city_match"]

    for model_name, predicted_values in predictions.items():
        predicted = np.asarray(predicted_values).reshape(-1)
        if len(predicted) != len(actual):
            raise ValueError(f"{model_name}: prediction length does not match y_val.")
        if not set(np.unique(predicted)).issubset({0, 1}):
            raise ValueError(f"{model_name}: predictions must contain only 0 and 1.")

        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        summary_rows.append({
            "model": model_name,
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
            "f1": f1_score(actual, predicted, zero_division=0),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        })

        # A confusion matrix shows the four types of predictions.
        matrix = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], ["Not buy (0)", "Buy (1)"])
        ax.set_yticks([0, 1], ["Not buy (0)", "Buy (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in model_name)
        fig.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=150)
        plt.close(fig)

        # Reset indexes to match labels by row position, not by pandas index.
        errors = features.reset_index(drop=True).copy()
        errors["actual"] = actual
        errors["predicted"] = predicted
        errors["error"] = actual != predicted
        errors["false_positive"] = (actual == 0) & (predicted == 1)
        errors["false_negative"] = (actual == 1) & (predicted == 0)

        for column in group_columns:
            if column not in errors.columns:
                continue
            grouped = (errors.groupby(column, dropna=False)
                       .agg(records=("error", "size"),
                            errors=("error", "sum"),
                            error_rate=("error", "mean"),
                            false_positives=("false_positive", "sum"),
                            false_negatives=("false_negative", "sum"))
                       .reset_index())
            grouped.insert(0, "feature", column)
            grouped = grouped.rename(columns={column: "feature_value"})
            grouped.insert(0, "model", model_name)
            group_tables.append(grouped)

        summary = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    summary.to_csv(output_dir / "model_comparison.csv", index=False)

    if group_tables:
        group_results = pd.concat(group_tables, ignore_index=True)
        group_results.to_csv(output_dir / "error_by_group.csv", index=False)
    else:
        group_results = pd.DataFrame()

    return summary, group_results
