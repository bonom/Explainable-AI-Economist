"""
Abstract algorithm class

Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""

from abc import ABC, abstractmethod
from typing import Tuple

from src.common.rollout_buffer import RolloutBuffer


class Model(ABC):
    """
    Abstract algorithm class
    """

    def __init__(
        self,
        observation_space,
        action_space,
    ):
        super().__init__()
        # Initialization
        # Environment and PPO parameters
        self.observation_space = observation_space
        self.action_space = action_space  # self.env.action_space.n

    @abstractmethod
    def act(self, observation: dict):
        """
        Return policy actions
        """
        NotImplementedError("This method must be implemented")

    @abstractmethod
    def learn(
        self,
        rollout_buffer: RolloutBuffer,
    ) -> Tuple[float, float]:
        """
        Update the policy
        """
        NotImplementedError("This method must be implemented")

    @abstractmethod
    def get_weights(self):
        """
        Get policy weights.

        Return:
            weights
        """
        NotImplementedError("This method must be implemented")

    @abstractmethod
    def set_weights(self, weights) -> None:
        """
        Set policy weights.
        """
        NotImplementedError("This method must be implemented")

    def save_model(self, name):
        """
        Save policy's model.
        """
        NotImplementedError("This method must be implemented")

    def load_model(self, name):
        """
        Load policy's model.
        """
        NotImplementedError("This method must be implemented")
