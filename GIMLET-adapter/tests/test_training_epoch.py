import copy
from unittest.mock import Mock

import pytest
import torch

from training_epoch import train_labeled_epoch


class ToyModel(torch.nn.Module):
    device = torch.device('cpu')

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, input_ids, labels):
        return {'loss': (self.layer(input_ids) - labels).square().mean()}


@pytest.mark.parametrize('n', [33, 146, 256])
def test_partial_windows_match_sample_weighted_large_batches(n):
    torch.manual_seed(6)
    model = ToyModel()
    reference = copy.deepcopy(model)
    x, y = torch.randn(n, 2), torch.randn(n, 1)
    loader = [{'input_ids': x[i:i+16], 'labels': y[i:i+16]} for i in range(0, n, 16)]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.step = Mock(wraps=optimizer.step)
    other = torch.optim.SGD(reference.parameters(), lr=.01)
    for epoch in range(2):
        train_labeled_epoch(model, loader, optimizer, 8)
        for i in range(0, n, 128):
            other.zero_grad()
            reference(x[i:i+128], y[i:i+128])['loss'].backward()
            other.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
    assert optimizer.step.call_count == 2 * ((n + 127) // 128)
