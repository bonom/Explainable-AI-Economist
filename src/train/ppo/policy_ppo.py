"""
Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""

from typing import List, Tuple

import torch

from src.common import Model
from src.train.ppo import LSTMModel
from src.common.rollout_buffer import SingleBuffer
from src.train.ppo.utils.execution_time import exec_time
from src.train.ppo import PytorchLinearA, PytorchLinearP


class PpoPolicy(Model):
    """
    PPO Main Optimization Algorithm
    """

    def __init__(
        self,
        observation_space,
        action_space,
        K_epochs: int = 16,
        eps_clip: int = 0.1,
        gamma: float = 0.998,
        c1: float = 0.5,
        c2: float = 0.01,
        learning_rate: float = 0.0003,
        device: str = 'cpu',
        model_type: str = 'linear',
        name: str = None,  # as an "a" or a "p"
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        # update policy for K epochs in one PPO update
        self.k_epochs = K_epochs
        # Clip parameter for PPO
        self.eps_clip = eps_clip
        # discount factor
        self.gamma = gamma
        # Hyperparameters in loss
        self._c1, self._c2 = c1, c2

        self.device = device
        # Environment and PPO parameters

        if model_type == 'linear':
            _type = PytorchLinearA if name == "a" else PytorchLinearP

            self.model: _type = _type(
                obs_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                learning_rate=learning_rate,
            ).to(self.device)
        elif model_type == 'lstm':
            self.model: LSTMModel = LSTMModel(
                obs_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
            )

        self.mse_loss = torch.nn.MSELoss()
        self.name = name

    def act(self, observation: dict):
        """
        Given an observation, returns `policy_action`, `policy_probability` and `vf_action` from the model.
        In this case (PPO) it's just a reference to call the model's forward method ->
        it's an "exposed API": common named functions for each policy.
        Args:
            observation: single agent observation of the environment.
        Returns:
            policy_action: predicted action(s)
            policy_probability: action probabilities
            vf_action: value function action predicted
        """
        # Get the prediction from the Actor network
        with torch.no_grad():
            policy_action, policy_probability = self.model.act(observation)

        return policy_action, policy_probability

    @exec_time
    def learn(
        self,
        rollout_buffer: SingleBuffer,
    ) -> dict:
        """
        Train Policy networks
        Takes as input the batch with N epochs of M steps_per_epoch. As we are using an LSTM
        model we are not shuffling all the data to create the minibatch, but only shuffling
        each epoch.
        Example:
            Input epochs: 0,1,2,3
            Shuffled epochs: 2,0,1,3
        It calls `self.Model.fit` passing the shuffled epoch.
        Args:
            rollout_buffer: RolloutBuffer for this specific policy.
        """

        data = self.__update(buffer=rollout_buffer)
        return data

    # @exec_time
    def __update(self, buffer: SingleBuffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(
            reversed(buffer.rewards), reversed(buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = rewards.reshape(*rewards.shape, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = buffer.states
        old_actions = buffer.actions
        old_logprobs = buffer.logprobs

        a_loss, c_loss, ac_entropy = [], [], []

        # Optimize policy for K epochs
        for i in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.model.evaluate(
                old_states[i:i+200:1], old_actions[i:i+200:1]
            )

            # Finding the ratio pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs[i:i+200:1])

            # Finding Surrogate Loss
            advantages = rewards[i:i+200:1] - state_values

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            critic_loss = self.mse_loss(state_values, rewards[i:i+200:1])

            # final loss of clipped objective PPO w/AI-Economist hyperparameters
            actor_loss = -torch.min(surr1, surr2)

            loss:torch.Tensor = actor_loss + self._c1 * critic_loss - self._c2 * dist_entropy

            # take gradient step
            self.model.optimizer.zero_grad()
            loss.mean().backward()
            self.model.optimizer.step()

            c_loss.append(torch.mean(critic_loss))
            a_loss.append(torch.mean(actor_loss))
            ac_entropy.append(torch.mean(dist_entropy))

        # self.model.scheduler.step()
        a_loss = torch.mean(torch.tensor(a_loss)).numpy()
        c_loss = torch.mean(torch.tensor(c_loss)).numpy()
        ac_entropy = torch.mean(torch.tensor(ac_entropy)).numpy()

        return {
            "a_loss": a_loss,
            "c_loss": c_loss,
            "entropy": ac_entropy,
            "loss": loss.mean().detach().numpy().item(), 
            "lr": self.model.optimizer.param_groups[0]["lr"],
        }

    def get_weights(self) -> dict:
        """
        Get policy weights.

        Return:
            actor_weights, critic_weights
        """
        actor_weights, critic_weights, optimizer_weights = self.model.get_weights()
        return {"a": actor_weights, "c": critic_weights, "o": optimizer_weights}

    def set_weights(self, weights: dict) -> None:
        """
        Set policy weights.
        """

        self.model.set_weights(
            actor_weights=weights["a"],
            critic_weights=weights["c"],
            optimizer_weights=weights["o"],
        )

    def save_model(self, path):
        """
        Save policy's model.
        """
        torch.save(self.model, path)

    def load_model(self, path):
        """
        Load policy's model.
        """
        self.model = torch.load(path)
