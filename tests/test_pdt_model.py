import numpy as np

from ml.pdt.model import PermutationDecisionTreeClassifier


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
