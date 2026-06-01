from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.features import FEATURE_COLUMNS, feature_matrix, load_ohlcv_csv, make_supervised_frame
from ml.pdt.model import PermutationDecisionTreeClassifier


@dataclass(frozen=True)
class TrainingConfig:
    csv_path: Path = Path("data/Binance_BTCUSDT_2026_minute.csv")
    model_path: Path = Path("models/pdt_btc_direction.joblib")
    metadata_path: Path = Path("models/pdt_btc_direction_metadata.json")
    train_tail_limit: int = 120_000
    n_trials: int = 24
    timeout_seconds: int = 55 * 60
    random_seed: int = 42
    horizon_candidates: tuple[int, ...] = (10, 15, 20, 30, 45, 60)
    etc_sample_limit: int = 256


@dataclass
class PreparedData:
    horizon_minutes: int
    train_frame: Any
    validation_frame: Any
    test_frame: Any
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    validation_baselines: dict[str, dict[str, Any]]


DEFAULT_CONFIG = TrainingConfig()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the production PDT model for BTC direction forecasting."
    )
    parser.parse_args()
    result = train_production_model(DEFAULT_CONFIG)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def train_production_model(config: TrainingConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    started_at = time.perf_counter()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    raw = load_ohlcv_csv(config.csv_path)
    if len(raw) > config.train_tail_limit:
        raw = raw.tail(config.train_tail_limit).reset_index(drop=True)

    train_raw, validation_raw, test_raw = temporal_split(raw)
    prepared_cache: dict[int, PreparedData] = {}

    def get_prepared(horizon_minutes: int) -> PreparedData:
        if horizon_minutes not in prepared_cache:
            prepared_cache[horizon_minutes] = prepare_data_for_horizon(
                train_raw,
                validation_raw,
                test_raw,
                horizon_minutes=horizon_minutes,
            )
        return prepared_cache[horizon_minutes]

    def objective(trial: optuna.Trial) -> float:
        horizon_minutes = trial.suggest_categorical(
            "horizon_minutes", list(config.horizon_candidates)
        )
        params = suggest_model_params(trial, config)
        prepared = get_prepared(horizon_minutes)

        trial_started_at = time.perf_counter()
        model = PermutationDecisionTreeClassifier(**params).fit(
            prepared.X_train,
            prepared.y_train,
            feature_names=FEATURE_COLUMNS,
        )
        predictions = model.predict_numeric(prepared.X_validation)
        confidence = model.predict_confidence(prepared.X_validation)
        metrics = evaluate_predictions(prepared.y_validation, predictions, confidence=confidence)
        score = objective_score(metrics, prepared.validation_baselines)

        trial.set_user_attr("validation_metrics", metrics)
        trial.set_user_attr("validation_baselines", prepared.validation_baselines)
        trial.set_user_attr("model_params", params)
        print(
            (
                f"trial={trial.number + 1} horizon={horizon_minutes} params={params} "
                f"score={score:.5f} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
                f"mean_confidence={metrics['confidence']['mean']:.4f} "
                f"coverage@0.55={metrics['confidence']['coverage_at_0.55']:.4f} "
                f"time={time.perf_counter() - trial_started_at:.1f}s"
            ),
            flush=True,
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.random_seed),
        study_name="pdt_btc_direction_production",
    )
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds)

    if study.best_trial is None:
        raise RuntimeError("Optuna не смогла подобрать PDT-параметры")

    selected_horizon = int(study.best_trial.params["horizon_minutes"])
    selected_params = dict(study.best_trial.user_attrs["model_params"])
    prepared = get_prepared(selected_horizon)

    X_train_final = np.vstack([prepared.X_train, prepared.X_validation])
    y_train_final = np.concatenate([prepared.y_train, prepared.y_validation])
    model = PermutationDecisionTreeClassifier(**selected_params).fit(
        X_train_final,
        y_train_final,
        feature_names=FEATURE_COLUMNS,
    )

    train_metrics = evaluate_model(model, X_train_final, y_train_final)
    test_metrics = evaluate_model(model, prepared.X_test, prepared.y_test)
    test_baselines = baseline_metrics(
        prepared.y_test,
        prepared.test_frame["previous_direction"].to_numpy(dtype=int),
        majority_prediction=majority_class(y_train_final),
    )

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "horizon_minutes": selected_horizon,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.model_path)

    metadata = {
        "model_name": "Permutation Decision Tree",
        "created_at": artifact["created_at"],
        "csv_path": str(config.csv_path),
        "model_path": str(config.model_path),
        "metadata_path": str(config.metadata_path),
        "train_tail_limit": config.train_tail_limit,
        "optimizer": {
            "name": "Optuna TPESampler",
            "n_trials": len(study.trials),
            "timeout_seconds": config.timeout_seconds,
            "best_value": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
        },
        "selected_params": selected_params,
        "horizon_minutes": selected_horizon,
        "horizon_candidates": list(config.horizon_candidates),
        "feature_columns": FEATURE_COLUMNS,
        "raw_rows": int(len(raw)),
        "split_rows": {
            "train": int(len(prepared.train_frame)),
            "validation": int(len(prepared.validation_frame)),
            "test": int(len(prepared.test_frame)),
        },
        "class_balance": {
            "train_final": class_balance(y_train_final),
            "validation": class_balance(prepared.y_validation),
            "test": class_balance(prepared.y_test),
        },
        "metrics": {
            "train": train_metrics,
            "validation": study.best_trial.user_attrs["validation_metrics"],
            "validation_baseline": study.best_trial.user_attrs["validation_baselines"],
            "test": test_metrics,
            "test_baseline": test_baselines,
        },
        "trial_history": trial_history(study),
        "elapsed_seconds": round(time.perf_counter() - started_at, 2),
    }
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "metadata": metadata,
        "summary": {
            "saved_model": str(config.model_path),
            "saved_metadata": str(config.metadata_path),
            "optimizer": metadata["optimizer"],
            "selected_horizon_minutes": selected_horizon,
            "selected_params": selected_params,
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_mean_confidence": test_metrics["confidence"]["mean"],
            "test_coverage_confidence_055": test_metrics["confidence"]["coverage_at_0.55"],
            "test_majority_baseline_accuracy": test_baselines["majority_class"]["accuracy"],
            "test_previous_direction_baseline_accuracy": test_baselines["previous_direction"]["accuracy"],
        },
    }


def suggest_model_params(trial: optuna.Trial, config: TrainingConfig) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_samples_leaf": trial.suggest_categorical(
            "min_samples_leaf", [25, 50, 100, 150, 250, 400]
        ),
        "max_thresholds": trial.suggest_categorical("max_thresholds", [16, 32, 64]),
        "etc_sample_limit": config.etc_sample_limit,
        "gini_weight": trial.suggest_categorical("gini_weight", [0.25, 0.5, 0.75, 1.0, 1.25]),
    }


def prepare_data_for_horizon(
    train_raw,
    validation_raw,
    test_raw,
    *,
    horizon_minutes: int,
) -> PreparedData:
    train_frame = make_supervised_frame(train_raw, horizon=horizon_minutes)
    validation_frame = make_supervised_frame(validation_raw, horizon=horizon_minutes)
    test_frame = make_supervised_frame(test_raw, horizon=horizon_minutes)
    ensure_non_empty_split(train_frame, validation_frame, test_frame)

    X_train = feature_matrix(train_frame)
    y_train = train_frame["target"].to_numpy(dtype=int)
    X_validation = feature_matrix(validation_frame)
    y_validation = validation_frame["target"].to_numpy(dtype=int)
    X_test = feature_matrix(test_frame)
    y_test = test_frame["target"].to_numpy(dtype=int)

    validation_baselines = baseline_metrics(
        y_validation,
        validation_frame["previous_direction"].to_numpy(dtype=int),
        majority_prediction=majority_class(y_train),
    )

    return PreparedData(
        horizon_minutes=horizon_minutes,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        validation_baselines=validation_baselines,
    )


def temporal_split(raw):
    train_end = int(len(raw) * 0.70)
    validation_end = int(len(raw) * 0.85)
    return (
        raw.iloc[:train_end].reset_index(drop=True),
        raw.iloc[train_end:validation_end].reset_index(drop=True),
        raw.iloc[validation_end:].reset_index(drop=True),
    )


def objective_score(
    metrics: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
) -> float:
    baseline_balanced_accuracy = max(
        0.5,
        *(baseline["balanced_accuracy"] for baseline in baselines.values()),
    )
    baseline_accuracy = max(baseline["accuracy"] for baseline in baselines.values())
    confidence = metrics["confidence"]
    prediction_bias = abs(
        metrics["prediction_balance"]["UP_share"] - metrics["class_balance"]["UP_share"]
    )

    balanced_edge = metrics["balanced_accuracy"] - baseline_balanced_accuracy
    confidence_edge = confidence["accuracy_at_0.55"] - baseline_accuracy
    confidence_term = confidence["coverage_at_0.55"] * confidence_edge
    bias_penalty = max(0.0, prediction_bias - 0.20)
    return float(balanced_edge + 0.25 * confidence_term - 0.05 * bias_penalty)


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    predictions = model.predict_numeric(X)
    confidence = model.predict_confidence(X)
    return evaluate_predictions(y, predictions, confidence=confidence)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray | None = None,
) -> dict[str, Any]:
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
        "class_balance": class_balance(y_true),
        "prediction_balance": class_balance(y_pred),
        "confidence": confidence_metrics(y_true, y_pred, confidence),
    }


def confidence_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray | None,
) -> dict[str, float]:
    if confidence is None:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "coverage_at_0.55": 0.0,
            "accuracy_at_0.55": 0.0,
            "coverage_at_0.60": 0.0,
            "accuracy_at_0.60": 0.0,
            "coverage_at_0.65": 0.0,
            "accuracy_at_0.65": 0.0,
        }

    values = np.asarray(confidence, dtype=float)
    result = {
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "p90": float(np.quantile(values, 0.90)) if len(values) else 0.0,
    }
    for threshold in (0.55, 0.60, 0.65):
        mask = values >= threshold
        suffix = f"{threshold:.2f}"
        result[f"coverage_at_{suffix}"] = float(mask.mean()) if len(mask) else 0.0
        result[f"accuracy_at_{suffix}"] = (
            float(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else 0.0
        )
    return result


def baseline_metrics(
    y_true: np.ndarray,
    previous_direction: np.ndarray,
    *,
    majority_prediction: int,
) -> dict[str, dict[str, Any]]:
    return {
        "majority_class": evaluate_predictions(
            y_true,
            np.full_like(y_true, fill_value=majority_prediction),
        ),
        "previous_direction": evaluate_predictions(
            y_true,
            previous_direction,
        ),
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


def trial_history(study: optuna.Study) -> list[dict[str, Any]]:
    trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    trials = sorted(trials, key=lambda item: item.value or -float("inf"), reverse=True)
    history: list[dict[str, Any]] = []
    for trial in trials[:10]:
        validation_metrics = trial.user_attrs.get("validation_metrics", {})
        history.append(
            {
                "number": int(trial.number),
                "value": float(trial.value),
                "params": trial.params,
                "validation_balanced_accuracy": validation_metrics.get("balanced_accuracy"),
                "validation_mean_confidence": validation_metrics.get("confidence", {}).get("mean"),
                "validation_coverage_confidence_055": validation_metrics.get("confidence", {}).get(
                    "coverage_at_0.55"
                ),
            }
        )
    return history


if __name__ == "__main__":
    main()
