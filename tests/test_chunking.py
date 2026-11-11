"""Unit tests for chunk_ranges — torch-free, runs in the lightweight CI."""
import pytest

from mcpy.utils.chunking import chunk_ranges


def test_chunk_size_one():
    assert list(chunk_ranges(3, 1)) == [(0, 1), (1, 2), (2, 3)]


def test_empty():
    assert list(chunk_ranges(0, 2)) == []


@pytest.mark.parametrize('bad', [0, -1])
def test_non_positive_chunk_is_single_range(bad):
    assert list(chunk_ranges(4, bad)) == [(0, 4)]
