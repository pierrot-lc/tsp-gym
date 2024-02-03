import pytest
import torch

from .tsp import random_instances


@pytest.mark.parametrize(
    "n_cities, n_instances, x_lim, y_lim",
    [
        (10, 100, (0, 1), (0, 1)),
        (10, 100, (0, 10), (0, 10)),
        (10, 100, (-10, 10), (-10, 10)),
        (10, 100, (-10, 10), (0, 1)),
        (10, 100, (0, 1), (-10, 10)),
    ],
)
def test_random_instances(
    n_cities: int, n_instances: int, x_lim: tuple[int, int], y_lim: tuple[int, int]
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    instances = random_instances(n_cities, n_instances, generator, x_lim, y_lim)

    assert instances.shape == torch.Size([n_instances, n_cities, 2])

    x_lim_test = (x_lim[0] <= instances[:, :, 0]) & (instances[:, :, 0] <= x_lim[1])
    y_lim_test = (y_lim[0] <= instances[:, :, 1]) & (instances[:, :, 1] <= y_lim[1])
    assert torch.all(x_lim_test)
    assert torch.all(y_lim_test)
