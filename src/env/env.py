import gymnasium as gym
import gymnasium.spaces as spaces
import torch

from .tsp import random_instances


class TSPEnv(gym):
    metadata = {"render.modes": ["computer"]}

    def __init__(self, instances: torch.Tensor, device: str, seed: int):
        """Initialize the environment with the given instances.

        ---
        Args:
            instances: The TSP instances.
                Shape of [batch_size, n_cities, 2].
            device: The device to use.
            seed: The seed to initialize the random number generator.
        """
        assert len(instances) == 3
        assert instances.shape[2] == 2

        super().__init__()
        self.instances = instances.to(device)
        self.device = device
        self.generator = torch.Generator(device).manual_seed(seed)
        self.batch_size, self.cities, _ = instances.shape

        # Spaces.
        self.node_space = spaces.Box(low=0, high=1, shape=(self.n_cities, 2))
        self.action_space = spaces.Discrete(self.cities)
        self.observation_space = self.node_space

        # Initialize dynamic infos.
        self.solutions = torch.zeros(
            (self.batch_size, self.cities), dtype=torch.long, device=device
        )
        self.current_city_id = 0

    def reset(self):
        self.instances = random_instances(self.batch_size, self.cities, self.generator)
        self.solutions.zero_()
        self.current_city_id = 0
