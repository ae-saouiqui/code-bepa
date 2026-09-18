import torch

from src.metrics.mlm import MLMMetrics


def test_perfect_predictions():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [0.0, 0.0, 10.0, 0.0],  # prediction: 2
            [0.0, 10.0, 0.0, 0.0],  # prediction: 1
            [0.0, 0.0, 0.0, 10.0],  # prediction: 3
            [10.0, 0.0, 0.0, 0.0],  # prediction: 0
        ]
    ])

    labels = torch.tensor([
        [2, 1, 3, 0]
    ])

    accuracy = metric(logits, labels)

    assert accuracy.item() == 1.0


def test_zero_correct_predictions():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [10.0, 0.0, 0.0, 0.0],  # prediction: 0
            [0.0, 0.0, 10.0, 0.0],  # prediction: 2
            [0.0, 10.0, 0.0, 0.0],  # prediction: 1
            [0.0, 0.0, 0.0, 10.0],  # prediction: 3
        ]
    ])

    labels = torch.tensor([
        [1, 3, 2, 0]
    ])

    accuracy = metric(logits, labels)

    assert accuracy.item() == 0.0


def test_partial_predictions():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [0.0, 10.0, 0.0, 0.0],  # prediction: 1 → correct
            [0.0, 0.0, 10.0, 0.0],  # prediction: 2 → wrong
            [0.0, 0.0, 0.0, 10.0],  # prediction: 3 → correct
            [10.0, 0.0, 0.0, 0.0],  # prediction: 0 → wrong
        ]
    ])

    labels = torch.tensor([
        [1, 3, 3, 2]
    ])

    accuracy = metric(logits, labels)

    assert accuracy.item() == 0.5


def test_ignores_non_masked_tokens():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [0.0, 10.0, 0.0],  # prediction: 1 → ignored
            [0.0, 10.0, 0.0],  # prediction: 1 → correct
            [0.0, 0.0, 10.0],  # prediction: 2 → ignored
            [10.0, 0.0, 0.0],  # prediction: 0 → correct
        ]
    ])

    labels = torch.tensor([
        [-100, 1, -100, 0]
    ])

    accuracy = metric(logits, labels)

    assert accuracy.item() == 1.0


def test_only_masked_tokens_are_used():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [10.0, 0.0, 0.0],  # prediction: 0 → ignored
            [0.0, 10.0, 0.0],  # prediction: 1 → correct
            [0.0, 0.0, 10.0],  # prediction: 2 → wrong
        ]
    ])

    labels = torch.tensor([
        [-100, 1, 0]
    ])

    accuracy = metric(logits, labels)

    assert accuracy.item() == 0.5


def test_multiple_sequences():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [0.0, 10.0, 0.0],  # prediction: 1
            [0.0, 0.0, 10.0],  # prediction: 2
        ],
        [
            [10.0, 0.0, 0.0],  # prediction: 0
            [0.0, 10.0, 0.0],  # prediction: 1
        ],
    ])

    labels = torch.tensor([
        [1, 0],  # one correct
        [0, 2],  # one correct
    ])

    accuracy = metric(logits, labels)

    assert accuracy.item() == 0.5


def test_returns_tensor():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [10.0, 0.0],
        ]
    ])

    labels = torch.tensor([
        [0]
    ])

    accuracy = metric(logits, labels)

    assert isinstance(accuracy, torch.Tensor)


def test_accuracy_is_between_zero_and_one():
    metric = MLMMetrics()

    logits = torch.tensor([
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    ])

    labels = torch.tensor([
        [0, 1, 2]
    ])

    accuracy = metric(logits, labels)

    assert 0.0 <= accuracy.item() <= 1.0

def test_unmasked_labels():
    metric = MLMMetrics()

    logits = torch.tensor([
        [10.0,2.0,4.0]
    ])

    labels = torch.tensor([
        [-100]
    ])

    accuracy = metric(logits,labels)

    assert accuracy == torch.tensor(0.0)
    