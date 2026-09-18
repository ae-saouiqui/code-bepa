import pytest
import torch

from src.models.bepa import CodeBEPA


MODEL_NAME = "microsoft/codebert-base"


@pytest.fixture
def model():
    return CodeBEPA(
        model_name=MODEL_NAME,
        hidden_size=64,
        output_dim=32,
    )


@pytest.fixture
def batch():
    batch_size = 2
    sequence_length = 16

    return {
        "input_ids": torch.randint(
            0,
            1000,
            (batch_size, sequence_length),
        ),

        "attention_mask": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.long,
        ),

        "mlm_input_ids": torch.randint(
            0,
            1000,
            (batch_size, sequence_length),
        ),

        "mlm_labels": torch.full(
            (batch_size, sequence_length),
            -100,
            dtype=torch.long,
        ),

        "masked_problems": torch.randint(
            0,
            1000,
            (batch_size, sequence_length),
        ),

        "masked_solutions": torch.randint(
            0,
            1000,
            (batch_size, sequence_length),
        ),

        "positions": [
            {
                "second_cls_pos": 8,
            },
            {
                "second_cls_pos": 9,
            },
        ],
    }


def test_model_initialization(model):
    assert isinstance(model, CodeBEPA)

    assert model.mlm_model is not None
    assert model.base_model is not None
    assert model.projector is not None
    assert model.predictor is not None


def test_encoder_is_shared(model):
    assert model.mlm_model.roberta is model.base_model


def test_forward_returns_four_outputs(model, batch):
    output = model(batch)

    assert len(output) == 4


def test_mlm_output_keys(model, batch):
    mlm_output, _, _, _ = model(batch)

    assert "loss" in mlm_output
    assert "logits" in mlm_output
    assert "hidden_states" in mlm_output


def test_mlm_loss_is_scalar(model, batch):
    mlm_output, _, _, _ = model(batch)

    assert mlm_output["loss"].ndim == 0


def test_mlm_logits_shape(model, batch):
    mlm_output, _, _, _ = model(batch)

    batch_size = batch["input_ids"].size(0)
    sequence_length = batch["input_ids"].size(1)
    vocab_size = model.mlm_model.config.vocab_size

    assert mlm_output["logits"].shape == (
        batch_size,
        sequence_length,
        vocab_size,
    )


def test_mlm_hidden_states_shape(model, batch):
    mlm_output, _, _, _ = model(batch)

    batch_size = batch["input_ids"].size(0)
    sequence_length = batch["input_ids"].size(1)
    hidden_size = model.base_model.config.hidden_size

    assert mlm_output["hidden_states"].shape == (
        batch_size,
        sequence_length,
        hidden_size,
    )


def test_problem_embedding_shape(model, batch):
    _, z_problem, _, _ = model(batch)

    batch_size = batch["input_ids"].size(0)
    output_dim = model.projector.mlp[-1].out_features

    assert z_problem.shape == (
        batch_size,
        output_dim,
    )


def test_solution_embedding_shape(model, batch):
    _, _, z_solution, _ = model(batch)

    batch_size = batch["input_ids"].size(0)
    output_dim = model.projector.mlp[-1].out_features

    assert z_solution.shape == (
        batch_size,
        output_dim,
    )


def test_predicted_solution_shape(model, batch):
    _, z_problem, _, predicted_solution = model(batch)

    assert predicted_solution.shape == z_problem.shape


def test_problem_and_solution_embeddings_same_shape(model, batch):
    _, z_problem, z_solution, _ = model(batch)

    assert z_problem.shape == z_solution.shape


def test_predictor_preserves_dimension(model, batch):
    _, z_problem, _, predicted_solution = model(batch)

    assert predicted_solution.size(-1) == z_problem.size(-1)


def test_different_cls_positions_are_supported(model, batch):
    _, _, z_solution, _ = model(batch)

    assert z_solution.size(0) == len(batch["positions"])
    