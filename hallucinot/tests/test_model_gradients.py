import torch
from torch import nn

from hallucinot.model import freeze_base_weights


def test_freezing_base_weights_preserves_activation_gradients():
    block = nn.Linear(3, 2)
    freeze_base_weights(block)
    activation = torch.ones((1, 3), requires_grad=True)

    output = block(activation)
    activation_gradient = torch.autograd.grad(output.sum(), activation)[0]

    assert not block.training
    assert all(not parameter.requires_grad for parameter in block.parameters())
    assert torch.isfinite(activation_gradient).all()
    assert torch.linalg.vector_norm(activation_gradient) > 0
    assert all(parameter.grad is None for parameter in block.parameters())
