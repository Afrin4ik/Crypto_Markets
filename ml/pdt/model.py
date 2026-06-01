from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ml.pdt.tree import TreeNode, build_tree, predict_node


@dataclass
class PermutationDecisionTreeClassifier:
    max_depth: int = 4
    min_samples_leaf: int = 250
    max_thresholds: int = 32
    etc_sample_limit: int | None = 256
    gini_weight: float = 0.75
    feature_names: list[str] = field(default_factory=list)
    root_: TreeNode | None = None
    classes_: tuple[int, int] = (0, 1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "PermutationDecisionTreeClassifier":
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        if features.ndim != 2:
            raise ValueError("X должен быть двумерной матрицей")
        if len(features) != len(labels):
            raise ValueError("Количество строк X и y должно совпадать")
        if len(features) == 0:
            raise ValueError("Нельзя обучить PDT на пустом датасете")

        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(features.shape[1])])
        self.root_ = build_tree(
            features,
            labels,
            depth=0,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_thresholds=self.max_thresholds,
            etc_sample_limit=self.etc_sample_limit,
            gini_weight=self.gini_weight,
        )
        return self

    def predict_numeric(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_leaf(row).prediction for row in self._as_matrix(X)], dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        numeric = self.predict_numeric(X)
        return np.where(numeric == 1, "UP", "DOWN")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = []
        for row in self._as_matrix(X):
            leaf = self._predict_leaf(row)
            probabilities.append([leaf.down_probability, leaf.up_probability])
        return np.asarray(probabilities, dtype=float)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        return np.max(self.predict_proba(X), axis=1)

    def _predict_leaf(self, row: np.ndarray):
        if self.root_ is None:
            raise ValueError("PDT-модель ещё не обучена")
        return predict_node(self.root_, row)

    @staticmethod
    def _as_matrix(X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            return features.reshape(1, -1)
        if features.ndim != 2:
            raise ValueError("X должен быть одномерным или двумерным массивом")
        return features
