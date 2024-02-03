import torch


def random_instances(
    n_cities: int,
    n_instances: int,
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
