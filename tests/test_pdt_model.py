import numpy as np
import pytest

from ml.pdt.model import PermutationDecisionTreeClassifier
from ml.pdt.tree import gini_gain, gini_impurity, quantile_thresholds


def test_pdt_predict_returns_direction_labels() -> None:
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = PermutationDecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=1,
        max_thresholds=4,
        etc_sample_limit=None,
    ).fit(X, y, feature_names=["x"])

    assert model.predict(np.array([[0.05], [0.95]])).tolist() == ["DOWN", "UP"]
    assert model.predict_proba(np.array([[0.95]])).shape == (1, 2)


def test_pdt_rejects_invalid_training_shapes() -> None:
    model = PermutationDecisionTreeClassifier()

    with pytest.raises(ValueError, match="двумерной"):
        model.fit(np.array([1, 2, 3]), np.array([0, 1, 0]))

    with pytest.raises(ValueError, match="совпадать"):
        model.fit(np.array([[1], [2]]), np.array([0]))


def test_pdt_predict_before_fit_raises_clear_error() -> None:
    model = PermutationDecisionTreeClassifier()

    with pytest.raises(ValueError, match="ещё не обучена"):
        model.predict(np.array([[0.5]]))


def test_gini_gain_is_positive_for_useful_split() -> None:
    labels = np.array([0, 0, 1, 1])
    left_mask = np.array([True, True, False, False])

    assert gini_impurity(labels) == 0.5
    assert gini_gain(labels, left_mask) == 0.5


def test_quantile_thresholds_stay_inside_observed_range() -> None:
    thresholds = quantile_thresholds(np.array([1, 2, 3, 4, 5], dtype=float), max_thresholds=2)

    assert np.allclose(
        thresholds,
        np.array([2.333333333333333, 3.6666666666666665]),
    )
