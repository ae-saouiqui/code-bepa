import torch

from src.losses.alignment import INFONCELoss


def test_returns_scalar_loss():
    loss_fn = INFONCELoss()

    originals = torch.randn(4, 128)
    projections = torch.randn(4, 128)

    loss = loss_fn(originals, projections)

    assert loss.ndim == 0


def test_loss_is_finite():
    loss_fn = INFONCELoss()

    originals = torch.randn(8, 128)
    projections = torch.randn(8, 128)

    loss = loss_fn(originals, projections)

    assert torch.isfinite(loss)


def test_loss_is_non_negative():
    loss_fn = INFONCELoss()

    originals = torch.randn(8, 128)
    projections = torch.randn(8, 128)

    loss = loss_fn(originals, projections)

    assert loss.item() >= 0.0


def test_matching_projections_have_lower_loss():
    loss_fn = INFONCELoss()

    originals = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])

    good_projections = originals.clone()

    bad_projections = torch.tensor([
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
    ])

    good_loss = loss_fn(originals, good_projections)
    bad_loss = loss_fn(originals, bad_projections)

    assert good_loss < bad_loss


def test_mismatched_projections_have_higher_loss():
    loss_fn = INFONCELoss()

    originals = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])

    good_projections = originals.clone()

    bad_projections = torch.tensor([
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
    ])

    good_loss = loss_fn(originals, good_projections)
    bad_loss = loss_fn(originals, bad_projections)

    assert good_loss < bad_loss


def test_labels_correspond_to_diagonal():
    loss_fn = INFONCELoss()

    originals = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    projections = originals.clone()

    loss = loss_fn(originals, projections)

    # With two perfectly matching pairs:
    #
    # similarity matrix:
    #
    # [[1, 0],
    #  [0, 1]]
    #
    # labels = [0, 1]
    #
    # Therefore the diagonal contains the positive pairs.
    expected = torch.nn.functional.cross_entropy(
        torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        torch.tensor([0, 1])
    )

    assert torch.allclose(loss, expected)


def test_different_batch_sizes():
    loss_fn = INFONCELoss()

    for batch_size in [2, 4, 8, 16]:
        originals = torch.randn(batch_size, 64)
        projections = torch.randn(batch_size, 64)

        loss = loss_fn(originals, projections)

        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_gradient_flows():
    loss_fn = INFONCELoss()

    originals = torch.randn(4, 32, requires_grad=True)
    projections = torch.randn(4, 32, requires_grad=True)

    loss = loss_fn(originals, projections)

    loss.backward()

    assert originals.grad is not None
    assert projections.grad is not None

    assert torch.isfinite(originals.grad).all()
    assert torch.isfinite(projections.grad).all()