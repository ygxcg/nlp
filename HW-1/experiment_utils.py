"""Shared LR selection, validation-only learning curves, and final evaluation."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split

SEED = 42
CLASSIFIER_SETTINGS = dict(solver="lbfgs", penalty="l2", max_iter=4000,
                           tol=1e-4, class_weight=None, random_state=SEED)
C_CANDIDATES = (0.1, 1.0, 10.0)
CURVE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(labels, predictions):
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0))}


def fit_classifier(features, labels, regularization):
    classifier = LogisticRegression(C=regularization, **CLASSIFIER_SETTINGS)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(features, labels)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return classifier, converged, round(time.perf_counter() - started, 3)


def select_and_evaluate(name, matrices, labels, details, artifacts: Path, stem):
    """Never consult test labels during C selection or learning curves."""
    train_x, valid_x, test_x = matrices
    train_y, valid_y, test_y = labels
    candidates, fitted = [], []
    for c in C_CANDIDATES:
        classifier, converged, seconds = fit_classifier(train_x, train_y, c)
        score = metrics(valid_y, classifier.predict(valid_x))
        candidates.append({"C": c, **score, "converged": converged,
                           "iterations": classifier.n_iter_.tolist(), "train_seconds": seconds})
        fitted.append(classifier)
        print(f"{name}: validation C={c:g}, macro-F1={score['macro_f1']:.4f}, "
              f"iterations={classifier.n_iter_.tolist()}, converged={converged}", flush=True)
    eligible = [i for i, item in enumerate(candidates) if item["converged"]]
    if not eligible:
        raise RuntimeError(f"No converged LR candidate for {name}")
    # Macro-F1 first, accuracy second, smaller C as deterministic tie-break.
    selected = max(eligible, key=lambda i: (candidates[i]["macro_f1"],
                   candidates[i]["accuracy"], -candidates[i]["C"]))
    classifier = fitted[selected]
    selected_c = candidates[selected]["C"]
    curve = []
    all_indices = np.arange(len(train_y))
    for fraction in CURVE_FRACTIONS:
        if fraction == 1.0:
            subset = all_indices
            model = classifier
            converged = True
        else:
            subset, _ = train_test_split(all_indices, train_size=fraction,
                                        stratify=train_y, random_state=SEED)
            model, converged, _ = fit_classifier(train_x[subset], train_y[subset], selected_c)
        if not converged:
            raise RuntimeError(f"Learning curve did not converge: {name}, fraction={fraction}")
        train_score = metrics(train_y[subset], model.predict(train_x[subset]))
        valid_score = metrics(valid_y, model.predict(valid_x))
        curve.append({"fraction": fraction, "train_documents": len(subset),
                      "train_indices_within_split": subset.tolist(),
                      "train_accuracy": train_score["accuracy"],
                      "train_macro_f1": train_score["macro_f1"],
                      "validation_accuracy": valid_score["accuracy"],
                      "validation_macro_f1": valid_score["macro_f1"],
                      "train_log_loss": float(log_loss(train_y[subset],
                          model.predict_proba(train_x[subset]), labels=model.classes_)),
                      "validation_log_loss": float(log_loss(valid_y,
                          model.predict_proba(valid_x), labels=model.classes_)),
                      "iterations": model.n_iter_.tolist(), "converged": converged})
        print(f"{name}: learning curve n={len(subset)}, "
              f"validation macro-F1={valid_score['macro_f1']:.4f}", flush=True)
    # This is the only test prediction/evaluation for the finalized candidate.
    valid_pred = classifier.predict(valid_x)
    test_pred = classifier.predict(test_x)
    valid_score = metrics(valid_y, valid_pred)
    test_score = metrics(test_y, test_pred)
    joblib.dump(classifier, artifacts / f"{stem}_classifier.joblib")
    result = {
        "representation": name, **details,
        "classifier": {**CLASSIFIER_SETTINGS, "C": selected_c},
        "selection_rule": "validation macro-F1, then accuracy, then smaller C; converged candidates only",
        "validation_candidates": candidates,
        "validation_accuracy": valid_score["accuracy"],
        "validation_macro_f1": valid_score["macro_f1"],
        "test_accuracy": test_score["accuracy"], "test_macro_f1": test_score["macro_f1"],
        "classifier_iterations": classifier.n_iter_.tolist(),
        "classifier_train_seconds": candidates[selected]["train_seconds"],
        "train_matrix_shape": list(train_x.shape), "classes": classifier.classes_.tolist(),
        "test_confusion_matrix": confusion_matrix(test_y, test_pred, labels=classifier.classes_).tolist(),
        "test_classification_report": classification_report(test_y, test_pred,
                                       output_dict=True, zero_division=0),
        "learning_curve_scope": "LR only; fixed features fit on full training texts; test excluded",
        "learning_curve": curve,
    }
    print(f"{name}: FINAL C={selected_c:g}, test accuracy={test_score['accuracy']:.4f}, "
          f"macro-F1={test_score['macro_f1']:.4f}", flush=True)
    return result, classifier, valid_pred, test_pred
