"""Unit tests for chunk_ranges — torch-free, runs in the lightweight CI."""
import pytest

from mcpy.utils.chunking import chunk_ranges


def test_none_yields_single_full_range():
    assert list(chunk_ranges(5, None)) == [(0, 5)]


def test_chunk_larger_than_n_is_single_range():
    assert list(chunk_ranges(3, 10)) == [(0, 3)]


def test_exact_division():
    assert list(chunk_ranges(6, 2)) == [(0, 2), (2, 4), (4, 6)]


def test_remainder_chunk():
    assert list(chunk_ranges(7, 3)) == [(0, 3), (3, 6), (6, 7)]


def test_chunk_size_one():
    assert list(chunk_ranges(3, 1)) == [(0, 1), (1, 2), (2, 3)]


def test_empty():
    assert list(chunk_ranges(0, 2)) == []


@pytest.mark.parametrize('bad', [0, -1])
def test_non_positive_chunk_is_single_range(bad):
    assert list(chunk_ranges(4, bad)) == [(0, 4)]
