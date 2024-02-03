from itertools import product

import pytest
import torch

from .tsp import compute_distances, evaluate_solutions, random_instances, sample_edges

generator = torch.Generator(device="cpu")
generator.manual_seed(0)


@pytest.mark.parametrize(
    "instances",
    [
        random_instances(10, 10, generator),
        random_instances(5, 10, generator),
        random_instances(50, 100, generator),
    ],
)
def test_evaluate_solutions(instances: torch.Tensor):
    n_instances, n_cities, _ = instances.shape
    solutions = [
        torch.randperm(n_cities, generator=generator) for _ in range(n_instances)
    ]
    solutions = torch.stack(solutions, dim=0)

    values = evaluate_solutions(instances, solutions)

    for instance, solution, value in zip(instances, solutions, values):
        total_distance = 0
        previous_city = solution[-1]
        for city in solution:
            total_distance += (
                ((instance[previous_city] - instance[city]) ** 2).sum().sqrt()
            )
            previous_city = city

        assert torch.allclose(value, total_distance)


@pytest.mark.parametrize(
    "instances, lambda_",
    [
        (random_instances(5, 10, generator), 1),
        (random_instances(5, 10, generator), 1),
        (random_instances(5, 10, generator), 1),
        (random_instances(5, 10, generator), 1),
    ],
)
def test_sample_edges(instances: torch.Tensor, lambda_: float):
    """Make sure each city is at least connected to its closest neighbour."""
    distances = compute_distances(instances)
    edges = sample_edges(distances, lambda_, generator)

    for edges_, distances_ in zip(edges, distances):
        for city_id in range(len(distances_)):
            distances_[city_id, city_id] = float("+inf")
            closest_city_id = torch.argmin(distances_[city_id])
            assert edges_[city_id, closest_city_id] == 1


@pytest.mark.parametrize(
    "instances",
    [
        random_instances(5, 10, generator),
        random_instances(5, 10, generator),
        random_instances(5, 10, generator),
        random_instances(5, 10, generator),
    ],
)
def test_compute_distances(instances: torch.Tensor):
    n_instances, n_cities, _ = instances.shape
    distances = compute_distances(instances)

    assert distances.shape == torch.Size([n_instances, n_cities, n_cities])

    ground_truth_distances = torch.zeros_like(distances)
    for instance_id, (instance, dists) in enumerate(zip(instances, distances)):
        for city_1, city_2 in product(range(n_cities), range(n_cities)):
            coords_1, coords_2 = instance[city_1], instance[city_2]
            dist = ((coords_1 - coords_2) ** 2).sum().sqrt()
            ground_truth_distances[instance_id, city_1, city_2] = dist

    assert torch.allclose(ground_truth_distances, distances)


@pytest.mark.parametrize(
    "n_cities, n_instances, x_lim, y_lim",
    [
        (100, 10, (0, 1), (0, 1)),
        (100, 10, (0, 10), (0, 10)),
        (100, 10, (-10, 10), (-10, 10)),
        (10, 100, (-10, 10), (0, 1)),
        (10, 100, (0, 1), (-10, 10)),
    ],
)
def test_random_instances(
    n_instances: int, n_cities: int, x_lim: tuple[int, int], y_lim: tuple[int, int]
):
    instances = random_instances(n_instances, n_cities, generator, x_lim, y_lim)

    assert instances.shape == torch.Size([n_instances, n_cities, 2])

    x_lim_test = (x_lim[0] <= instances[:, :, 0]) & (instances[:, :, 0] <= x_lim[1])
    y_lim_test = (y_lim[0] <= instances[:, :, 1]) & (instances[:, :, 1] <= y_lim[1])
    assert torch.all(x_lim_test)
    assert torch.all(y_lim_test)
