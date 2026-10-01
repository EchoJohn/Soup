"""#1528: literal token matching must not rescan a long needle at every word."""

from __future__ import annotations

import random
import re
import time
from typing import Callable

import pytest

from soup_cli.eval.custom import score_contains

_SCALE_N = 2_500
_SCALE_LIMIT = 24.0
_INPUTS = {
    "repeated-words-miss": lambda n: ("a " * n, "a " * (n // 2) + "b"),
    "repeated-words-final-match": lambda n: ("a " * n + "b", "a " * (n // 2) + "b"),
    "inside-one-token": lambda n: ("a" * (2 * n), "a" * n),
    "overlapping-boundary-misses": lambda n: ("a-" * n, "-a" * (n // 2)),
}


def _best_time(pair: tuple[str, str], repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        score_contains(*pair)
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("build", list(_INPUTS.values()), ids=list(_INPUTS))
def test_contains_grows_linearly(build: Callable[[int], tuple[str, str]]) -> None:
    small, large = build(_SCALE_N), build(8 * _SCALE_N)
    # Warm both compiled patterns so the ratio measures the scan itself.
    score_contains(*small)
    score_contains(*large)
    t_small = max(_best_time(small, repeats=5), 1e-7)
    t_large = float("inf")
    for _ in range(5):
        t_large = min(t_large, _best_time(large, repeats=1))
        if t_large / t_small < _SCALE_LIMIT:
            break
    assert t_large / t_small < _SCALE_LIMIT, (t_small, t_large)


def test_contains_agrees_with_previous_regex_on_ascii() -> None:
    rng = random.Random(1528)
    alphabet = "aAbBcC019 .-#+(|"
    for index in range(20_000):
        output = "".join(rng.choices(alphabet, k=rng.randrange(50)))
        if index % 2 and output:
            start = rng.randrange(len(output))
            expected = output[start:rng.randrange(start, len(output) + 1)].swapcase()
        else:
            expected = "".join(rng.choices(alphabet, k=rng.randrange(12)))
        needle = expected.strip()
        pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
        previous = not needle or re.search(pattern, output, re.IGNORECASE) is not None
        assert score_contains(output, expected) is previous, (output, expected)


@pytest.mark.parametrize(
    ("output", "expected", "matched"),
    [
        ("İstanbul", "istanbul", False),
        ("ſ", "s", False),
        ("ÉCOLE", "école", True),
        ("a\nb", "A\nB", True),
        ("xa\nb", "a\nb", False),
        ("xa-a-a ", "a-a", True),  # The valid occurrence overlaps a rejected one.
        ("x" * 60_000 + " answer: gold.", "gold", True),
    ],
    ids=["dotted-i", "long-s", "accent", "multiline", "left-boundary", "overlap", "full-output"],
)
def test_contains_literal_and_boundary_cases(output: str, expected: str, matched: bool) -> None:
    assert score_contains(output, expected) is matched
