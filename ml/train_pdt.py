from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.features import FEATURE_COLUMNS, feature_matrix, load_ohlcv_csv, make_supervised_frame
from ml.pdt.model import PermutationDecisionTreeClassifier


DEFAULT_CSV_PATH = Path("data/Binance_BTCUSDT_2026_minute.csv")
DEFAULT_MODEL_PATH = Path("models/pdt_btc_direction.joblib")
DEFAULT_METADATA_PATH = Path("models/pdt_btc_direction_metadata.json")


def main() -> None:
    args = parse_args()
    result = train_from_args(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PDT model for BTC direction forecasting.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-tail-limit", type=int, default=120_000)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=250)
    parser.add_argument("--max-thresholds", type=int, default=32)
    parser.add_argument("--etc-sample-limit", type=int, default=256)
    parser.add_argument("--no-grid", action="store_true", help="Skip validation grid search.")
    return parser.parse_args()


def train_from_args(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_ohlcv_csv(args.csv_path)
    if args.train_tail_limit and len(raw) > args.train_tail_limit:
        raw = raw.tail(args.train_tail_limit).reset_index(drop=True)

    train_raw, validation_raw, test_raw = temporal_split(raw)
    train_frame = make_supervised_frame(train_raw, horizon=args.horizon)
    validation_frame = make_supervised_frame(validation_raw, horizon=args.horizon)
    test_frame = make_supervised_frame(test_raw, horizon=args.horizon)

    X_train = feature_matrix(train_frame)
    y_train = train_frame["target"].to_numpy(dtype=int)
    X_validation = feature_matrix(validation_frame)
    y_validation = validation_frame["target"].to_numpy(dtype=int)
    X_test = feature_matrix(test_frame)
    y_test = test_frame["target"].to_numpy(dtype=int)

    ensure_non_empty_split(train_frame, validation_frame, test_frame)

    best_params, validation_metrics = select_params(
        X_train,
        y_train,
        X_validation,
        y_validation,
        args=args,
    )

    X_train_final = np.vstack([X_train, X_validation])
    y_train_final = np.concatenate([y_train, y_validation])
    model = PermutationDecisionTreeClassifier(**best_params).fit(
        X_train_final,
        y_train_final,
        feature_names=FEATURE_COLUMNS,
    )

    train_metrics = evaluate_predictions(y_train_final, model.predict_numeric(X_train_final))
    test_predictions = model.predict_numeric(X_test)
    test_metrics = evaluate_predictions(y_test, test_predictions)
    baseline_metrics = {
        "majority_class": evaluate_predictions(
            y_test,
            np.full_like(y_test, fill_value=majority_class(y_train_final)),
        ),
        "previous_direction": evaluate_predictions(
            y_test,
            test_frame["previous_direction"].to_numpy(dtype=int),
        ),
    }

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "horizon_minutes": args.horizon,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path)

    metadata = {
        "model_name": "Permutation Decision Tree",
        "created_at": artifact["created_at"],
        "csv_path": str(args.csv_path),
        "model_path": str(args.model_path),
        "horizon_minutes": args.horizon,
        "train_tail_limit": args.train_tail_limit,
        "feature_columns": FEATURE_COLUMNS,
        "selected_params": best_params,
        "raw_rows": int(len(raw)),
        "split_rows": {
            "train": int(len(train_frame)),
            "validation": int(len(validation_frame)),
            "test": int(len(test_frame)),
        },
        "class_balance": {
            "train_final": class_balance(y_train_final),
            "validation": class_balance(y_validation),
            "test": class_balance(y_test),
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "baseline": baseline_metrics,
        },
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "metadata": metadata,
        "summary": {
            "saved_model": str(args.model_path),
            "saved_metadata": str(args.metadata_path),
            "selected_params": best_params,
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "baseline_majority_accuracy": baseline_metrics["majority_class"]["accuracy"],
            "baseline_previous_direction_accuracy": baseline_metrics["previous_direction"]["accuracy"],
        },
    }


def temporal_split(raw):
    train_end = int(len(raw) * 0.70)
    validation_end = int(len(raw) * 0.85)
    return (
        raw.iloc[:train_end].reset_index(drop=True),
        raw.iloc[train_end:validation_end].reset_index(drop=True),
        raw.iloc[validation_end:].reset_index(drop=True),
    )


def select_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.no_grid:
        candidates = [
            {
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "max_thresholds": args.max_thresholds,
                "etc_sample_limit": args.etc_sample_limit,
            }
        ]
    else:
        candidates = [
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_thresholds": max_thresholds,
                "etc_sample_limit": args.etc_sample_limit,
            }
            for max_depth, min_samples_leaf, max_thresholds in product(
                [3, 4, 5],
                [250, 500],
                [16, 32],
            )
        ]

    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None

    for params in candidates:
        model = PermutationDecisionTreeClassifier(**params).fit(
            X_train,
            y_train,
            feature_names=FEATURE_COLUMNS,
        )
        predictions = model.predict_numeric(X_validation)
        metrics = evaluate_predictions(y_validation, predictions)
        score = (metrics["balanced_accuracy"], metrics["f1"], metrics["accuracy"])
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("Не удалось подобрать параметры PDT")
    return best_params, best_metrics


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int).tolist(),
    }


def class_balance(y: np.ndarray) -> dict[str, Any]:
    total = len(y)
    down = int((y == 0).sum())
    up = int((y == 1).sum())
    return {
        "DOWN": down,
        "UP": up,
        "DOWN_share": down / total if total else 0.0,
        "UP_share": up / total if total else 0.0,
    }


def majority_class(y: np.ndarray) -> int:
    return 1 if int((y == 1).sum()) > int((y == 0).sum()) else 0


def ensure_non_empty_split(*frames) -> None:
    for name, frame in zip(("train", "validation", "test"), frames):
        if frame.empty:
            raise ValueError(f"После feature engineering split {name} оказался пустым")


if __name__ == "__main__":
    main()
