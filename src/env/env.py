import gymnasium as gym
import gymnasium.spaces as spaces
import torch
from einops import repeat

import itertools

from .tsp import random_instances, evaluate_solutions


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
        self.partial_solutions = torch.zeros(
            (self.batch_size, self.cities), dtype=torch.long, device=device
        )
        self.partial_solutions.fill_(-1)
        self.current_step = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )

    def reset(self):
        self.instances = random_instances(self.batch_size, self.cities, self.generator)
        self.partial_solutions.fill_(-1)
        self.current_step.zero_()

    def step(
        self, city_ids: torch.Tensor
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str],
    ]:
        assert torch.all(self.current_step < self.cities)
        assert torch.all(
            self.partial_solutions != repeat(city_ids, "b -> b c", c=self.cities)
        ), "Some city is already choosen in the current partial solutions."

        self.partial_solutions[
            torch.arange(self.batch_size, device=self.device), self.current_step
        ] = city_ids
        self.current_step += 1

        dones = self.current_step == self.cities
        rewards = torch.zeros(self.batch_size, device=self.device)
        if torch.any(dones):
            rewards[dones] = evaluate_solutions(
                self.instances[dones], self.partial_solutions[dones]
            )
        truncated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        infos = {}

        return self.render(), rewards, dones, truncated, infos

    def render(self):
        return self.instances, self.partial_solutions
