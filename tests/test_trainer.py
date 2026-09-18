"""
This test script was written py ChatGPT 
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.trainer import CodeBEPATrainer


# ============================================================
# Tiny model used only for testing the Trainer
# ============================================================

class TinyCodeBEPA(nn.Module):

    def __init__(self, hidden_dim=8, vocab_size=20):
        super().__init__()

        self.encoder = nn.Linear(4, hidden_dim)
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch):

        x = batch["features"]

        hidden = self.encoder(x)

        logits = self.mlm_head(hidden)

        labels = batch["mlm_labels"]

        mlm_loss = nn.functional.cross_entropy(
            logits,
            labels
        )

        mlm_output = {
            "loss": mlm_loss,
            "logits": logits
        }

        z_problem = self.projector(hidden)

        z_solution = self.projector(
            batch["solution_features"]
        )

        predicted_solution = self.predictor(
            z_problem
        )

        return (
            mlm_output,
            z_problem,
            z_solution,
            predicted_solution
        )


# ============================================================
# Test data helpers
# ============================================================

def create_batch(batch_size=4):

    return {
        "features": torch.randn(
            batch_size,
            4
        ),

        "solution_features": torch.randn(
            batch_size,
            8
        ),

        "mlm_labels": torch.randint(
            0,
            20,
            (batch_size,)
        )
    }


def create_dataloader(
    number_of_batches=4,
    batch_size=4
):

    batches = [
        create_batch(batch_size)
        for _ in range(number_of_batches)
    ]

    return DataLoader(
        batches,
        batch_size=None
    )


def create_trainer(
    train_batches=4,
    val_batches=2,
    batch_size=4,
    accumulation_step=1
):

    model = TinyCodeBEPA()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    train_loader = create_dataloader(
        number_of_batches=train_batches,
        batch_size=batch_size
    )

    val_loader = create_dataloader(
        number_of_batches=val_batches,
        batch_size=batch_size
    )

    trainer = CodeBEPATrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        alignement_weight=1.0,
        device="cpu",
        accumulation_step=accumulation_step
    )

    return trainer


# ============================================================
# train_epoch()
# ============================================================

def test_train_epoch_returns_expected_metrics():

    trainer = create_trainer()

    result = trainer.train_epoch()

    expected_keys = {
        "mlm_loss",
        "alignement_loss",
        "total_loss",
        "mlm_accuracy",
        "infonce_accuracy"
    }

    assert set(result.keys()) == expected_keys


def test_train_epoch_returns_finite_metrics():

    trainer = create_trainer()

    result = trainer.train_epoch()

    for value in result.values():

        assert torch.isfinite(
            torch.tensor(value)
        )


def test_train_epoch_updates_model_parameters():

    trainer = create_trainer()

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    trainer.train_epoch()

    after = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    changed = any(
        not torch.equal(
            before[name],
            after[name]
        )
        for name in before
    )

    assert changed


# ============================================================
# validate()
# ============================================================

def test_validate_returns_expected_metrics():

    trainer = create_trainer()

    result = trainer.validate()

    expected_keys = {
        "mlm_loss",
        "alignement_loss",
        "total_loss",
        "mlm_accuracy",
        "infonce_accuracy",
        "problem_rankme",
        "solution_rankme"
    }

    assert set(result.keys()) == expected_keys


def test_validate_returns_finite_metrics():

    trainer = create_trainer()

    result = trainer.validate()

    for value in result.values():

        assert torch.isfinite(
            torch.tensor(value)
        )


def test_validate_does_not_update_model():

    trainer = create_trainer()

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    trainer.validate()

    after = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    for name in before:

        assert torch.equal(
            before[name],
            after[name]
        )


def test_validate_sets_model_to_eval_mode():

    trainer = create_trainer()

    trainer.model.train()

    trainer.validate()

    assert not trainer.model.training


# ============================================================
# Optimizer step
# ============================================================

def test_optimizer_step_clears_gradients():

    trainer = create_trainer()

    trainer.optimizer.zero_grad()

    batch = create_batch()

    output = trainer.model(batch)

    mlm_loss = output[0]["loss"]

    alignment_loss = trainer.alignement_loss(
        output[3],
        output[2]
    )

    loss = (
        mlm_loss
        + alignment_loss
    )

    # IMPORTANT:
    # Use GradScaler exactly as the real training loop does.
    trainer.scaler.scale(loss).backward()

    has_gradients = any(
        parameter.grad is not None
        for parameter in trainer.model.parameters()
    )

    assert has_gradients

    trainer._optimizer_step()

    for parameter in trainer.model.parameters():

        assert parameter.grad is None


# ============================================================
# Gradient accumulation
# ============================================================

def test_gradient_accumulation_updates_model():

    trainer = create_trainer(
        train_batches=4,
        accumulation_step=2
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    trainer.train_epoch()

    after = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    changed = any(
        not torch.equal(
            before[name],
            after[name]
        )
        for name in before
    )

    assert changed


def test_gradient_accumulation_discards_leftover_batches():

    trainer = create_trainer(
        train_batches=5,
        accumulation_step=2
    )

    optimizer_steps = 0

    original_optimizer_step = (
        trainer._optimizer_step
    )

    def counted_optimizer_step():

        nonlocal optimizer_steps

        optimizer_steps += 1

        original_optimizer_step()

    trainer._optimizer_step = (
        counted_optimizer_step
    )

    trainer.train_epoch()

    # 5 batches / accumulation of 2
    #
    # Steps:
    #   batch 1 + 2 -> optimizer step
    #   batch 3 + 4 -> optimizer step
    #   batch 5     -> discarded
    #
    assert optimizer_steps == 2


# ============================================================
# Checkpointing
# ============================================================

def test_save_checkpoint_creates_file(tmp_path):

    trainer = create_trainer()

    trainer.epoch = 1

    trainer.save_checkpoint(
        str(tmp_path)
    )

    checkpoint_path = (
        tmp_path / "epoch_1.pt"
    )

    assert checkpoint_path.exists()


def test_checkpoint_contains_expected_keys(tmp_path):

    trainer = create_trainer()

    trainer.epoch = 1

    trainer.save_checkpoint(
        str(tmp_path)
    )

    checkpoint = torch.load(
        tmp_path / "epoch_1.pt",
        map_location="cpu"
    )

    expected_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict"
    }

    assert set(checkpoint.keys()) == expected_keys


def test_checkpoint_contains_correct_epoch(tmp_path):

    trainer = create_trainer()

    trainer.epoch = 7

    trainer.save_checkpoint(
        str(tmp_path)
    )

    checkpoint = torch.load(
        tmp_path / "epoch_7.pt",
        map_location="cpu"
    )

    assert checkpoint["epoch"] == 7


def test_checkpoint_model_state_dict_is_dictionary(tmp_path):

    trainer = create_trainer()

    trainer.epoch = 1

    trainer.save_checkpoint(
        str(tmp_path)
    )

    checkpoint = torch.load(
        tmp_path / "epoch_1.pt",
        map_location="cpu"
    )

    assert isinstance(
        checkpoint["model_state_dict"],
        dict
    )


def test_checkpoint_can_restore_model(tmp_path):

    trainer = create_trainer()

    trainer.train_epoch()

    trainer.epoch = 1

    trainer.save_checkpoint(
        str(tmp_path)
    )

    checkpoint = torch.load(
        tmp_path / "epoch_1.pt",
        map_location="cpu"
    )

    new_model = TinyCodeBEPA()

    new_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    for original, restored in zip(
        trainer.model.parameters(),
        new_model.parameters()
    ):

        assert torch.equal(
            original,
            restored
        )


# ============================================================
# train()
# ============================================================

def test_train_returns_history(tmp_path):

    trainer = create_trainer(
        train_batches=2,
        val_batches=1
    )

    original_save_checkpoint = (
        trainer.save_checkpoint
    )

    def save_checkpoint():

        original_save_checkpoint(
            str(tmp_path)
        )

    trainer.save_checkpoint = (
        save_checkpoint
    )

    history = trainer.train(
        epochs=2
    )

    assert len(history) == 2

    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2


def test_train_saves_checkpoint_before_validation(
    tmp_path
):

    trainer = create_trainer(
        train_batches=2,
        val_batches=1
    )

    original_save_checkpoint = (
        trainer.save_checkpoint
    )

    def save_checkpoint():

        original_save_checkpoint(
            str(tmp_path)
        )

    trainer.save_checkpoint = (
        save_checkpoint
    )

    def failing_validation():

        raise RuntimeError(
            "Intentional validation failure"
        )

    trainer.validate = (
        failing_validation
    )

    try:

        trainer.train(
            epochs=1
        )

    except RuntimeError as error:

        assert str(error) == (
            "Intentional validation failure"
        )

    checkpoint_path = (
        tmp_path / "epoch_1.pt"
    )

    assert checkpoint_path.exists()