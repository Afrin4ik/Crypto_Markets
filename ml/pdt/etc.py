from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable

import numpy as np


def calculate_etc(sequence: Iterable[int]) -> int:
    """Calculate Effort-To-Compress for a symbolic sequence."""
    seq = [int(item) for item in sequence]
    if len(seq) <= 1 or len(set(seq)) <= 1:
        return 0

    iterations = 0
    max_symbol = max(seq)

    while len(set(seq)) > 1:
        pair_counts = Counter(zip(seq, seq[1:]))
        if not pair_counts:
            break
        highest_freq_pair = pair_counts.most_common(1)[0][0]
        max_symbol += 1
        new_symbol = max_symbol

        compressed: list[int] = []
        index = 0
        while index < len(seq):
            if index < len(seq) - 1 and (seq[index], seq[index + 1]) == highest_freq_pair:
                compressed.append(new_symbol)
                index += 2
            else:
                compressed.append(seq[index])
                index += 1

        if len(compressed) == len(seq):
            break
        seq = compressed
        iterations += 1

    return iterations


def etc_score(sequence: Iterable[int], sample_limit: int | None = None) -> int:
    sampled = _sample_sequence(sequence, sample_limit)
    return _calculate_etc_cached(sampled)


def etc_gain(labels: np.ndarray, left_mask: np.ndarray, sample_limit: int | None = None) -> float:
    left_labels = labels[left_mask]
    right_labels = labels[~left_mask]
    if len(left_labels) == 0 or len(right_labels) == 0:
        return -float("inf")

    total_etc = etc_score(labels, sample_limit=sample_limit)
    left_etc = etc_score(left_labels, sample_limit=sample_limit)
    right_etc = etc_score(right_labels, sample_limit=sample_limit)
    weighted_etc = (len(left_labels) / len(labels)) * left_etc + (
        len(right_labels) / len(labels)
    ) * right_etc
    return float(total_etc - weighted_etc)


@lru_cache(maxsize=4096)
def _calculate_etc_cached(sequence: tuple[int, ...]) -> int:
    return calculate_etc(sequence)


def _sample_sequence(sequence: Iterable[int], sample_limit: int | None) -> tuple[int, ...]:
    values = np.asarray(list(sequence), dtype=int)
    if sample_limit is None or sample_limit <= 0 or len(values) <= sample_limit:
        return tuple(int(item) for item in values)
    indices = np.linspace(0, len(values) - 1, num=sample_limit, dtype=int)
    return tuple(int(item) for item in values[indices])
