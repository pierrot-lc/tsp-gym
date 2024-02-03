import torch


def sample_edges(
    distances: torch.Tensor, lambda_: float, generator: torch.Generator
) -> torch.Tensor:
    """Sample edges based on the distance between each cities.
    The closer two cities are, the more likely an edge will exist.
    Each city is at least connected to its closest city.

    The probabilities are computed based on the exponential distribution.

    ---
    Args:
        distances: The distances between each city.
            Shape of [n_instances, n_cities, n_cities].
        lambda_: The lambda parameter of the exponential distribution.
        generator: The random number generator.

    ---
    Returns:
        The adjacency matrix.
            Shape of [n_instances, n_cities, n_cities].
    """
    device = distances.device
    arange = torch.arange(distances.shape[1], device=device)

    # Sample the edges.
    probabilities = (1 / lambda_) * torch.exp(-distances / lambda_)
    edges = torch.rand(probabilities.shape, generator=generator) <= probabilities
    edges = edges.int()

    # Remove the self-loops.
    edges[:, arange, arange] = 0

    # Get the closest city.
    distances = distances.clone()
    distances[:, arange, arange] = float("+inf")
    closest_cities = torch.argmin(distances, dim=2)

    # Connect each city to its closest city.
    edges.scatter_(2, closest_cities.unsqueeze(2), 1)

    return edges

def compute_distances(instances: torch.Tensor) -> torch.Tensor:
    """Compute the distances between each city.

    ---
    Args:
        instances: The TSP instances.
            Shape of [n_instances, n_cities, 2].

    ---
    Returns:
        The distance matrix.
            Shape of [n_instances, n_cities, n_cities].
    """
    return torch.cdist(instances, instances, p=2)


def random_instances(
    n_instances: int,
    n_cities: int,
    generator: torch.Generator,
    x_lim: tuple[int, int] = (0, 1),
    y_lim: [int, int] = (0, 1),
) -> torch.Tensor:
    """Generate a batch of random instances where cities are drawn uniformly.

    ---
    Returns
        The TSP instances.
            Shape of [n_instances, n_cities, 2].
    """
    assert n_instances > 0
    assert n_cities > 1
    assert x_lim[0] <= x_lim[1]
    assert y_lim[0] <= y_lim[1]

    x = torch.rand(
        (n_instances, n_cities, 1),
        generator=generator,
        device=generator.device,
    )
    y = torch.rand(
        (n_instances, n_cities, 1),
        generator=generator,
        device=generator.device,
    )

    x = x * (x_lim[1] - x_lim[0]) + x_lim[0]
    y = y * (y_lim[1] - y_lim[0]) + y_lim[0]

    return torch.cat([x, y], dim=2)
