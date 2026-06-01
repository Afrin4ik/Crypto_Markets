from ml.pdt.etc import calculate_etc


def test_calculate_etc_uniform_sequence_is_zero() -> None:
    assert calculate_etc([1, 1, 1, 1]) == 0


def test_calculate_etc_mixed_sequence_is_positive() -> None:
    assert calculate_etc([0, 1, 0, 1, 1, 0]) > 0
