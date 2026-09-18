import pytest
import torch
from src.data.collators import CodeBEPADataCollator




# Creating a fake tokenizer 
class FakeTokenizer:

    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1
    mask_token_id = 99
    vocab_size = 1000

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        # Deterministic IDs for testing
        return [
            i + 10
            for i in range(len(tokens))
        ]


# Fixtures

@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def collator(tokenizer):
    return CodeBEPADataCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_length=20
    )


@pytest.fixture
def batch():
    return [
        (
            "sort an array",
            "def sort_array return sorted"
        ),
        (
            "find maximum value",
            "def maximum return max"
        )
    ]


# Output

def test_collator_returns_expected_keys(
    collator,
    batch
):
    output = collator(batch)

    expected_keys = {
        "input_ids",
        "attention_mask",
        "mlm_input_ids",
        "mlm_labels",
        "masked_problems",
        "masked_solutions",
        "positions"
    }

    assert set(output.keys()) == expected_keys


# Batch size

def test_collator_preserves_batch_size(
    collator,
    batch
):
    output = collator(batch)

    assert output["input_ids"].shape[0] == len(batch)
    assert output["attention_mask"].shape[0] == len(batch)
    assert output["mlm_input_ids"].shape[0] == len(batch)
    assert output["mlm_labels"].shape[0] == len(batch)
    assert output["masked_problems"].shape[0] == len(batch)
    assert output["masked_solutions"].shape[0] == len(batch)

    assert len(output["positions"]) == len(batch)


# Shapes

def test_collator_output_shapes(
    collator,
    batch
):
    output = collator(batch)

    input_shape = output["input_ids"].shape

    assert output["attention_mask"].shape == input_shape
    assert output["mlm_input_ids"].shape == input_shape
    assert output["mlm_labels"].shape == input_shape
    assert output["masked_problems"].shape == input_shape
    assert output["masked_solutions"].shape == input_shape


# Dtype

def test_collator_returns_long_tensors(
    collator,
    batch
):
    output = collator(batch)

    assert output["input_ids"].dtype == torch.long
    assert output["attention_mask"].dtype == torch.long
    assert output["mlm_input_ids"].dtype == torch.long
    assert output["mlm_labels"].dtype == torch.long
    assert output["masked_problems"].dtype == torch.long
    assert output["masked_solutions"].dtype == torch.long


# Original sequence structure
# It has to [CLS]Porblem [SEP][CLS]Solution[SEP][PAD]...[PAD]

def test_sequence_structure(
    collator,
    batch
):
    output = collator(batch)

    input_ids = output["input_ids"]
    positions = output["positions"]

    tokenizer = collator.tokenizer

    for i, pos in enumerate(positions):

        assert (
            input_ids[i, pos["first_cls_pos"]]
            == tokenizer.cls_token_id
        )

        assert (
            input_ids[i, pos["first_sep_pos"]]
            == tokenizer.sep_token_id
        )

        assert (
            input_ids[i, pos["second_cls_pos"]]
            == tokenizer.cls_token_id
        )

        assert (
            input_ids[i, pos["second_sep_pos"]]
            == tokenizer.sep_token_id
        )


# positions

def test_positions_are_consistent(
    collator,
    batch
):
    output = collator(batch)

    positions = output["positions"]

    for pos in positions:

        assert pos["first_cls_pos"] == 0

        assert (
            pos["problem_start"]
            <= pos["problem_end"]
        )

        assert (
            pos["first_sep_pos"]
            == pos["problem_end"] + 1
        )

        assert (
            pos["second_cls_pos"]
            == pos["first_sep_pos"] + 1
        )

        assert (
            pos["solution_start"]
            == pos["second_cls_pos"] + 1
        )

        assert (
            pos["solution_start"]
            <= pos["solution_end"]
        )

        assert (
            pos["second_sep_pos"]
            == pos["solution_end"] + 1
        )


# Attention Masks

def test_attention_mask(
    collator,
    batch
):
    output = collator(batch)

    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]

    pad_id = collator.tokenizer.pad_token_id

    for i in range(len(batch)):

        for j in range(input_ids.shape[1]):

            if input_ids[i, j] == pad_id:
                assert attention_mask[i, j] == 0
            else:
                assert attention_mask[i, j] == 1


# BEPA masking problems

def test_masked_problems(
    collator,
    batch
):
    output = collator(batch)

    original = output["input_ids"]
    masked = output["masked_problems"]
    positions = output["positions"]

    mask_id = collator.tokenizer.mask_token_id

    for i, pos in enumerate(positions):

        # First CLS remains visible
        assert (
            masked[i, pos["first_cls_pos"]]
            == original[i, pos["first_cls_pos"]]
        )

        # Problem + first SEP are masked
        for p in range(
            pos["problem_start"],
            pos["first_sep_pos"] + 1
        ):
            assert masked[i, p] == mask_id

        # Second CLS remains visible
        assert (
            masked[i, pos["second_cls_pos"]]
            == original[i, pos["second_cls_pos"]]
        )

        # Solution remains untouched
        for p in range(
            pos["solution_start"],
            pos["second_sep_pos"] + 1
        ):
            assert (
                masked[i, p]
                == original[i, p]
            )


# Masking solutions

def test_masked_solutions(
    collator,
    batch
):
    output = collator(batch)

    original = output["input_ids"]
    masked = output["masked_solutions"]
    positions = output["positions"]

    mask_id = collator.tokenizer.mask_token_id

    for i, pos in enumerate(positions):

        # First CLS remains visible
        assert (
            masked[i, pos["first_cls_pos"]]
            == original[i, pos["first_cls_pos"]]
        )

        # Problem + first SEP remain untouched
        for p in range(
            pos["problem_start"],
            pos["first_sep_pos"] + 1
        ):
            assert (
                masked[i, p]
                == original[i, p]
            )

        # Second CLS + solution + second SEP are masked
        for p in range(
            pos["second_cls_pos"],
            pos["second_sep_pos"] + 1
        ):
            assert masked[i, p] == mask_id


# MLM labels

def test_mlm_labels(
    collator,
    batch
):
    output = collator(batch)

    labels = output["mlm_labels"]
    positions = output["positions"]

    vocab_size = collator.tokenizer.vocab_size

    # Labels must either be -100 or valid token IDs
    assert torch.all(
        (labels == -100)
        | (
            (labels >= 0)
            & (labels < vocab_size)
        )
    )

    # Special tokens must never have MLM labels
    for i, pos in enumerate(positions):

        special_positions = [
            pos["first_cls_pos"],
            pos["first_sep_pos"],
            pos["second_cls_pos"],
            pos["second_sep_pos"],
        ]

        for p in special_positions:
            assert labels[i, p] == -100


# Input

def test_mlm_input_preserves_shape(
    collator,
    batch
):
    output = collator(batch)

    assert (
        output["mlm_input_ids"].shape
        == output["input_ids"].shape
    )


# Test if original input is modified 

def test_original_input_is_not_masked(
    collator,
    batch
):
    output = collator(batch)

    input_ids = output["input_ids"]
    positions = output["positions"]

    mask_id = collator.tokenizer.mask_token_id

    for i, pos in enumerate(positions):

        for p in range(pos["problem_start"],pos["problem_end"] + 1):
            assert input_ids[i, p] != mask_id

        for p in range(pos["solution_start"],pos["solution_end"] + 1):
            assert input_ids[i, p] != mask_id