"""Task 1: shared whitespace vocabulary, Binary and Frequency."""
from __future__ import annotations
import argparse
import csv
import hashlib
import platform
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from experiment_utils import CLASSIFIER_SETTINGS, C_CANDIDATES, SEED, select_and_evaluate, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "HW-1" / "nyt.csv"
DEFAULT_OUTPUT = ROOT / "task1_results.json"


def load_data(path):
    texts, labels = [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("NYT requires text and label columns")
        for row in reader:
            text, label = (row.get("text") or "").strip(), (row.get("label") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, np.asarray(labels)


def make_split(texts, labels):
    indices = np.arange(len(texts))
    train_idx, heldout_idx = train_test_split(indices, test_size=0.2,
        random_state=SEED, stratify=labels)
    valid_idx, test_idx = train_test_split(heldout_idx, test_size=0.5,
        random_state=SEED, stratify=labels[heldout_idx])
    return train_idx, valid_idx, test_idx


def whitespace_tokens(text):
    # Keep punctuation, numbers, stopwords and single-character words.
    return text.lower().split()


def explain_document(matrix, classifier, index, vocabulary, true_label, wrong_label):
    row = matrix[index].tocsr()
    true_id = list(classifier.classes_).index(true_label)
    wrong_id = list(classifier.classes_).index(wrong_label)
    differences = classifier.coef_[true_id] - classifier.coef_[wrong_id]
    contributions = row.data * differences[row.indices]
    terms = [{"token": vocabulary[column], "feature_value": float(value),
              "true_minus_wrong_contribution": float(effect)}
             for column, value, effect in zip(row.indices, row.data, contributions)]
    positive = sorted((t for t in terms if t["true_minus_wrong_contribution"] > 0),
                      key=lambda t: -t["true_minus_wrong_contribution"])[:8]
    negative = sorted((t for t in terms if t["true_minus_wrong_contribution"] < 0),
                      key=lambda t: t["true_minus_wrong_contribution"])[:8]
    return {"compared_labels": [true_label, wrong_label],
            "intercept_difference": float(classifier.intercept_[true_id] - classifier.intercept_[wrong_id]),
            "logit_margin_true_minus_wrong": float(row.dot(differences)[0]
                + classifier.intercept_[true_id] - classifier.intercept_[wrong_id]),
            "supports_true": positive, "supports_wrong": negative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "task1_artifacts")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    texts, labels = load_data(args.data)
    indices = make_split(texts, labels)
    names = ("train", "validation", "test")
    documents = [[texts[i] for i in subset] for subset in indices]
    targets = [labels[subset] for subset in indices]
    write_json(args.artifacts / "split_indices.json",
               {name: subset.tolist() for name, subset in zip(names, indices)})
    # Build the vocabulary once. Binary is derived from the same count matrices.
    vectorizer = CountVectorizer(analyzer=whitespace_tokens, lowercase=False, dtype=np.float64)
    count_matrices = [vectorizer.fit_transform(documents[0]),
                      vectorizer.transform(documents[1]), vectorizer.transform(documents[2])]
    binary_matrices = [matrix.copy() for matrix in count_matrices]
    for matrix in binary_matrices:
        matrix.data.fill(1.0)
    joblib.dump(vectorizer, args.artifacts / "count_vectorizer.joblib")
    feature_names = vectorizer.get_feature_names_out()
    write_json(args.artifacts / "vocabulary.json", vectorizer.vocabulary_)
    experiments = [("Binary Bag-of-Words", "binary", binary_matrices, "required"),
                   ("Word Frequency", "frequency", count_matrices, "required")]
    results, models, predictions, validation_predictions = [], {}, {}, {}
    with threadpool_limits(limits=1):
        for name, stem, matrices, role in experiments:
            result, model, valid_pred, test_pred = select_and_evaluate(name, matrices, targets,
                {"vocabulary_size": len(feature_names), "role": role,
                 "tokenization": "text.lower().split(); punctuation/numbers/stopwords retained"},
                args.artifacts, stem)
            results.append(result)
            models[stem], predictions[stem] = model, test_pred
            validation_predictions[stem] = valid_pred
    for split_name, subset, truth, prediction_map in [
        ("test", indices[2], targets[2], predictions),
        ("validation", indices[1], targets[1], validation_predictions)]:
        with (args.artifacts / f"{split_name}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["representation", "nyt_row_index", "true_label", "predicted_label"])
            for name, stem, _, _ in experiments:
                writer.writerows(zip([name] * len(subset), subset.tolist(), truth.tolist(),
                                     prediction_map[stem].tolist()))
    bad_cases = []
    for i, (truth, binary, frequency) in enumerate(zip(targets[2], predictions["binary"], predictions["frequency"])):
        if binary == truth and frequency == truth:
            continue
        wrong_label = frequency if frequency != truth else binary
        tokens = whitespace_tokens(documents[2][i])
        bad_cases.append({
            "nyt_row_index": int(indices[2][i]), "test_position": i, "true_label": str(truth),
            "binary_prediction": str(binary), "frequency_prediction": str(frequency),
            "group": "binary_only_correct" if binary == truth else
                     "frequency_only_correct" if frequency == truth else "both_wrong",
            "text": documents[2][i], "token_count": len(tokens),
            "most_repeated_tokens": Counter(tokens).most_common(15),
            "binary_evidence": explain_document(binary_matrices[2], models["binary"], i,
                                               feature_names, truth, wrong_label),
            "frequency_evidence": explain_document(count_matrices[2], models["frequency"], i,
                                                  feature_names, truth, wrong_label),
        })
    write_json(args.artifacts / "bad_cases.json", bad_cases)
    data_digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    output = {
        "data": str(args.data), "data_sha256": data_digest, "random_seed": SEED,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "split": {name: len(subset) for name, subset in zip(names, indices)},
        "split_label_distribution": {name: dict(Counter(labels[subset].tolist()))
                                    for name, subset in zip(names, indices)},
        "label_distribution": dict(Counter(labels.tolist())),
        "preprocessing": {"lowercase": True, "tokenization": "whitespace split",
                          "remove_punctuation": False, "remove_stopwords": False,
                          "remove_numbers": False, "truncate": False, "vocabulary_scope": "training only"},
        "model": CLASSIFIER_SETTINGS, "C_candidates": list(C_CANDIDATES),
        "bad_case_counts": dict(Counter(item["group"] for item in bad_cases)),
        "experiments": results,
    }
    write_json(args.output, output)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
