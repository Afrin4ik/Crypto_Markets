from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.pdt.etc import etc_gain


@dataclass
class LeafStats:
    prediction: int
    n_samples: int
    down_probability: float
    up_probability: float
    confidence: float


@dataclass
class TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    leaf: LeafStats | None = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None


@dataclass
class SplitCandidate:
    feature_index: int
    threshold: float
    gain: float
    etc_gain: float
    gini_gain: float


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_thresholds: int,
    etc_sample_limit: int | None,
    gini_weight: float,
) -> TreeNode:
    y = y.astype(int)
    if _should_stop(y, depth, max_depth, min_samples_leaf):
        return TreeNode(leaf=_leaf_stats(y))

    split = find_best_split(
        X,
        y,
        min_samples_leaf=min_samples_leaf,
        max_thresholds=max_thresholds,
        etc_sample_limit=etc_sample_limit,
        gini_weight=gini_weight,
    )
    if split is None or split.gain <= 0:
        return TreeNode(leaf=_leaf_stats(y))

    left_mask = X[:, split.feature_index] <= split.threshold
    return TreeNode(
        feature_index=split.feature_index,
        threshold=split.threshold,
        left=build_tree(
            X[left_mask],
            y[left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_thresholds=max_thresholds,
            etc_sample_limit=etc_sample_limit,
            gini_weight=gini_weight,
        ),
        right=build_tree(
            X[~left_mask],
            y[~left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_thresholds=max_thresholds,
            etc_sample_limit=etc_sample_limit,
            gini_weight=gini_weight,
        ),
    )


def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    min_samples_leaf: int,
    max_thresholds: int,
    etc_sample_limit: int | None,
    gini_weight: float,
) -> SplitCandidate | None:
    best: SplitCandidate | None = None
    parent_gini = gini_impurity(y)

    for feature_index in range(X.shape[1]):
        values = X[:, feature_index]
        thresholds = quantile_thresholds(values, max_thresholds=max_thresholds)
        for threshold in thresholds:
            left_mask = values <= threshold
            left_count = int(left_mask.sum())
            right_count = len(y) - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue

            split_etc_gain = etc_gain(y, left_mask, sample_limit=etc_sample_limit)
            split_gini_gain = gini_gain(y, left_mask, parent_gini=parent_gini)
            normalized_etc_gain = split_etc_gain / max(len(y), 1)
            gain = normalized_etc_gain + (gini_weight * split_gini_gain)
            if best is None or gain > best.gain:
                best = SplitCandidate(
                    feature_index=feature_index,
                    threshold=float(threshold),
                    gain=float(gain),
                    etc_gain=float(split_etc_gain),
                    gini_gain=float(split_gini_gain),
                )

    return best


def gini_impurity(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    down_probability = float((y == 0).sum() / len(y))
    up_probability = 1.0 - down_probability
    return 1.0 - down_probability**2 - up_probability**2


def gini_gain(y: np.ndarray, left_mask: np.ndarray, *, parent_gini: float | None = None) -> float:
    left_labels = y[left_mask]
    right_labels = y[~left_mask]
    if len(left_labels) == 0 or len(right_labels) == 0:
        return -float("inf")
    parent = gini_impurity(y) if parent_gini is None else parent_gini
    weighted_children = (len(left_labels) / len(y)) * gini_impurity(left_labels) + (
        len(right_labels) / len(y)
    ) * gini_impurity(right_labels)
    return parent - weighted_children


def quantile_thresholds(values: np.ndarray, max_thresholds: int) -> np.ndarray:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size < 2:
        return np.array([], dtype=float)

    unique = np.unique(finite)
    if unique.size < 2:
        return np.array([], dtype=float)

    if unique.size <= max_thresholds + 1:
        thresholds = (unique[:-1] + unique[1:]) / 2.0
    else:
        quantiles = np.linspace(0, 1, num=max_thresholds + 2)[1:-1]
        thresholds = np.quantile(finite, quantiles)

    thresholds = np.unique(thresholds)
    return thresholds[(thresholds > unique[0]) & (thresholds < unique[-1])]


def predict_node(node: TreeNode, row: np.ndarray) -> LeafStats:
    current = node
    while not current.is_leaf:
        if current.feature_index is None or current.threshold is None:
            raise ValueError("Некорректный узел PDT без feature_index/threshold")
        current = current.left if row[current.feature_index] <= current.threshold else current.right
        if current is None:
            raise ValueError("Некорректное дерево PDT: отсутствует дочерний узел")
    if current.leaf is None:
        raise ValueError("Некорректное дерево PDT: отсутствует leaf")
    return current.leaf


def _should_stop(y: np.ndarray, depth: int, max_depth: int, min_samples_leaf: int) -> bool:
    return (
        len(y) == 0
        or np.unique(y).size == 1
        or depth >= max_depth
        or len(y) < min_samples_leaf * 2
    )


def _leaf_stats(y: np.ndarray) -> LeafStats:
    if len(y) == 0:
        return LeafStats(0, 0, 1.0, 0.0, 1.0)
    down_count = int((y == 0).sum())
    up_count = int((y == 1).sum())
    n_samples = int(len(y))
    down_probability = down_count / n_samples
    up_probability = up_count / n_samples
    prediction = 1 if up_probability > down_probability else 0
    confidence = max(down_probability, up_probability)
    return LeafStats(
        prediction=prediction,
        n_samples=n_samples,
        down_probability=down_probability,
        up_probability=up_probability,
        confidence=confidence,
    )
