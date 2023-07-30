"""
Empty policy to manage an `unmanaged` policy. Used when an actor (or set of actors)
is not under a real policy. Every action is set to 0.

Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=import-error

from typing import Tuple

import numpy as np

from src.common import Model
from src.common.rollout_buffer import RolloutBuffer


class EmptyModel(Model):
    """
    Empty policy to manage an `unmanaged` policy. Used when an actor (or set of actors)
    is not under a real policy. Every action is set to 0.
    """

    def __init__(self, observation_space, action_space):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

    def act(self, observation: dict):
        """
        Returns this policy actions and a useless placeholder here only to do
        not manage in a smart way all the multi-agent-policy system.
        """
        actions = []
        # TODO: revert
        # for _ in self.action_space:
        #   actions.append(torch.zeros((1,)))

        actions = np.zeros(self.action_space)

        return actions, np.zeros(1)

    def learn(self, rollout_buffer: RolloutBuffer) -> Tuple[float, float]:
        """
        This policy doesn't have to learn anything. It will just do nothing.
        """
        return {
            "a_loss": 0.0,
            "c_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0, 
            "lr": 0.0,
        }

    def get_weights(self):
        """
        Get policy weights.

        Return:
            weights
        """
        return {"a": None, "c": None}

    def set_weights(self, weights) -> None:
        """
        Set policy weights.

        Return:
            weights
        """

    def save_model(self, name):
        """
        Save policy's model.
        """

    def load_model(self, name):
        """
        Load policy's model.
        """
