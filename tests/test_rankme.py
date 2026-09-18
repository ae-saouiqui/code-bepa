"""
The test is written using ChatGPT
"""


import pytest
import torch

from src.metrics.rankme import RankMEMetric


@pytest.fixture
def rankme():
    return RankMEMetric()


def test_rankme_full_rank_embeddings(rankme):
    """
    Four orthogonal directions with equal singular values
    should have effective rank close to 4.
    """
    embeddings = torch.eye(4)

    rank = rankme(embeddings)

    assert rank == pytest.approx(4.0, abs=1e-5)


def test_rankme_rank_one_embeddings(rankme):
    """
    All samples lie in the same direction.
    Effective rank should be close to 1.
    """
    embeddings = torch.tensor([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ])

    rank = rankme(embeddings)

    assert rank == pytest.approx(1.0, abs=1e-5)


def test_rankme_collapsed_embeddings(rankme):
    """
    Completely identical embeddings represent a collapsed space.
    """
    embeddings = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])

    rank = rankme(embeddings)

    assert rank == pytest.approx(1.0, abs=1e-5)


def test_rankme_is_bounded_by_matrix_rank(rankme):
    """
    Effective rank cannot exceed min(number_of_samples, embedding_dimension).
    """
    embeddings = torch.randn(10, 4)

    rank = rankme(embeddings)

    assert 1.0 <= rank <= 4.0


def test_rankme_returns_float(rankme):
    embeddings = torch.randn(8, 16)

    rank = rankme(embeddings)

    assert isinstance(rank, float)


def test_rankme_higher_for_diverse_embeddings(rankme):
    """
    A diverse/full-rank representation should have a higher
    effective rank than a collapsed representation.
    """
    collapsed = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])

    diverse = torch.eye(4, 3)

    collapsed_rank = rankme(collapsed)
    diverse_rank = rankme(diverse)

    assert diverse_rank > collapsed_rank