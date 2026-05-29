from __future__ import annotations
# Gigamit aron suportahan ang modern Python type hints bisan sa mas daan nga Python versions.

import numpy as np
# Gi-import ang NumPy library. Apan sa maong file wala kini magamit, busa pwede ni tangtangon aron limpyo ang code.

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
# Mga evaluation metrics gikan sa scikit-learn:
# accuracy_score  -> porsyento sa sakto nga prediction
# precision_score -> unsa ka sakto ang mga gi-predict nga spam
# recall_score    -> pila ka spam ang nadakpan sa model
# f1_score        -> kombinasyon sa precision ug recall
# confusion_matrix-> matrix nga nagpakita sa prediction results


def evaluate_model(y_true, y_pred):
    """
    Function nga mo-evaluate sa performance sa model.

    Parameters:
        y_true -> tinuod nga labels gikan sa dataset
        y_pred -> labels nga gi-predict sa model

    Returns:
        Dictionary nga adunay tanan evaluation metrics.
    """

    return {
        # Accuracy = overall correctness sa model
        "accuracy": float(accuracy_score(y_true, y_pred)),

        # Precision = sa tanan giingong spam, pila ang tinuod nga spam
        "precision": float(
            precision_score(y_true, y_pred, pos_label="spam")
        ),

        # Recall = sa tanan tinuod nga spam, pila ang nadakpan sa model
        "recall": float(
            recall_score(y_true, y_pred, pos_label="spam")
        ),

        # F1 Score = balance sa precision ug recall
        "f1_score": float(
            f1_score(y_true, y_pred, pos_label="spam")
        ),

        # Confusion Matrix:
        # [[ham->ham, ham->spam],
        #  [spam->ham, spam->spam]]
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=["ham", "spam"]
        ).tolist(),
    }


def compare_models(results):
    """
    Function nga mo-compare sa daghang trained models.

    Parameters:
        results -> dictionary nga adunay metrics sa matag model

    Example:
    {
        "Logistic Regression": {
            "metrics": {
                "f1_score": 0.95
            }
        },
        "Naive Bayes": {
            "metrics": {
                "f1_score": 0.91
            }
        }
    }

    Returns:
        String summary sa best-performing model.
    """

    # I-sort ang tanang models base sa F1 Score gikan sa pinakataas ngadto sa pinakagamay.
    ranked = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["f1_score"],
        reverse=True
    )

    # Kuhaon ang pinakaunang item human sa sorting
    # mao ni ang best model
    best_name, best_result = ranked[0]

    # Mobalik ug formatted summary
    return (
        f"Best model: {best_name} | "
        f"Accuracy: {best_result['metrics']['accuracy']:.3f} | "
        f"Precision: {best_result['metrics']['precision']:.3f} | "
        f"Recall: {best_result['metrics']['recall']:.3f} | "
        f"F1: {best_result['metrics']['f1_score']:.3f}"
    )