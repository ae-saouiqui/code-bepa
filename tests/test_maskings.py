import pytest
import torch
from dataclasses import dataclass
from src.data.maskings import (
    mlm_masking,
    mask_problems,
    mask_solutions,
)


# Creating necceary data for the test 
@dataclass
class FakeTokenizer:
    mask_token_id = 99
    vocab_size = 1000


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def sequences():
    # [CLS1] P1 P2 [SEP1] [CLS2] S1 S2 [SEP2]
    return torch.tensor([
        [0, 10, 11, 2, 0, 20, 21, 2],
        [0, 30, 31, 2, 0, 40, 41, 2],
    ])


@pytest.fixture
def positions():
    return [
        {
            "first_cls_pos": 0,
            "problem_start": 1,
            "problem_end": 2,
            "first_sep_pos": 3,
            "second_cls_pos": 4,
            "solution_start": 5,
            "solution_end": 6,
            "second_sep_pos": 7,
        },
        {
            "first_cls_pos": 0,
            "problem_start": 1,
            "problem_end": 2,
            "first_sep_pos": 3,
            "second_cls_pos": 4,
            "solution_start": 5,
            "solution_end": 6,
            "second_sep_pos": 7,
        },
    ]

# Testing mlm masking

def test_mlm_zero_probability(
    sequences,
    positions,
    tokenizer
):
    masked, labels = mlm_masking(
        sequences,
        positions,
        tokenizer,
        pourcentage=0.0
    )

    # Nothing should be masked
    assert torch.equal(masked, sequences)

    # No prediction targets
    assert torch.all(labels == -100)


def test_mlm_probability_one(
    sequences,
    positions,
    tokenizer
):
    masked, labels = mlm_masking(
        sequences,
        positions,
        tokenizer,
        pourcentage=1.0
    )

    for i in range(len(sequences)):

        # Problem tokens must have labels
        for pos in range(
            positions[i]["problem_start"],
            positions[i]["problem_end"] + 1
        ):
            assert labels[i, pos] == sequences[i, pos]

        # Solution tokens must have labels
        for pos in range(
            positions[i]["solution_start"],
            positions[i]["solution_end"] + 1
        ):
            assert labels[i, pos] == sequences[i, pos]

        # Special tokens must not have labels
        assert labels[i, positions[i]["first_cls_pos"]] == -100
        assert labels[i, positions[i]["first_sep_pos"]] == -100
        assert labels[i, positions[i]["second_cls_pos"]] == -100
        assert labels[i, positions[i]["second_sep_pos"]] == -100


def test_mlm_does_not_modify_original(
    sequences,
    positions,
    tokenizer
):
    original = sequences.clone()

    mlm_masking(
        sequences,
        positions,
        tokenizer,
        pourcentage=1.0
    )

    assert torch.equal(sequences, original)


def test_mlm_output_shapes(
    sequences,
    positions,
    tokenizer
):
    masked, labels = mlm_masking(
        sequences,
        positions,
        tokenizer,
        pourcentage=0.15
    )

    assert masked.shape == sequences.shape
    assert labels.shape == sequences.shape


def test_mlm_invalid_probability(
    sequences,
    positions,
    tokenizer
):
    with pytest.raises(ValueError):
        mlm_masking(
            sequences,
            positions,
            tokenizer,
            pourcentage=1.5
        )


def test_mlm_invalid_probability_type(
    sequences,
    positions,
    tokenizer
):
    with pytest.raises(TypeError):
        mlm_masking(
            sequences,
            positions,
            tokenizer,
            pourcentage=1
        )


# problem masking

def test_mask_problems(
    sequences,
    positions,
    tokenizer
):
    masked = mask_problems(
        sequences,
        positions,
        tokenizer
    )

    for i in range(len(sequences)):

        # Problem tokens + first SEP are masked
        assert masked[i, 1] == tokenizer.mask_token_id
        assert masked[i, 2] == tokenizer.mask_token_id
        assert masked[i, 3] == tokenizer.mask_token_id

        # CLS1 remains
        assert masked[i, 0] == sequences[i, 0]

        # CLS2 remains
        assert masked[i, 4] == sequences[i, 4]

        # Solution remains
        assert torch.equal(
            masked[i, 5:7],
            sequences[i, 5:7]
        )

        # Second SEP remains
        assert masked[i, 7] == sequences[i, 7]


def test_mask_problems_does_not_modify_original(
    sequences,
    positions,
    tokenizer
):
    original = sequences.clone()

    mask_problems(
        sequences,
        positions,
        tokenizer
    )

    assert torch.equal(sequences, original)


# Solution masking

def test_mask_solutions(
    sequences,
    positions,
    tokenizer
):
    masked = mask_solutions(
        sequences,
        positions,
        tokenizer
    )

    for i in range(len(sequences)):

        # CLS1 remains
        assert masked[i, 0] == sequences[i, 0]

        # Problem remains
        assert torch.equal(
            masked[i, 1:3],
            sequences[i, 1:3]
        )

        # First SEP remains
        assert masked[i, 3] == sequences[i, 3]

        # CLS2 + solution + second SEP are masked
        assert masked[i, 4] == tokenizer.mask_token_id
        assert masked[i, 5] == tokenizer.mask_token_id
        assert masked[i, 6] == tokenizer.mask_token_id
        assert masked[i, 7] == tokenizer.mask_token_id


def test_mask_solutions_does_not_modify_original(
    sequences,
    positions,
    tokenizer
):
    original = sequences.clone()

    mask_solutions(
        sequences,
        positions,
        tokenizer
    )

    assert torch.equal(sequences, original)