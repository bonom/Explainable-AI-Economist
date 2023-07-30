"""
AI-Economist inspired pytorch linear model

Author: bonom
Github: https://github.com/bonom

Adapted from: Sa1g (https://github.com/sa1g)

Date: 2023-07-30

"""
import torch
import numpy as np
import torch.nn as nn
from gym.spaces import Box, Dict


def get_flat_obs_size(obs_space):
    """
    Get flat observation size
    """
    if isinstance(obs_space, Box):
        return np.prod(obs_space.shape)
    elif not isinstance(obs_space, Dict):
        raise TypeError

    def rec_size(obs_dict_space, n=0):
        for subspace in obs_dict_space.spaces.values():
            if isinstance(subspace, Box):
                n = n + np.prod(subspace.shape)
            elif isinstance(subspace, Dict):
                n = rec_size(subspace, n=n)
            else:
                raise TypeError
        return n

    return rec_size(obs_space)


def apply_logit_mask1(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = torch.ones(logits.shape) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask


class PytorchLinearA(nn.Module):
    """A linear (feed-forward) model."""

    def __init__(self, obs_space, action_space, device, learning_rate):
        super().__init__()
        self.device = device

        self.MASK_NAME = "action_mask"
        self.num_outputs = action_space
        self.logit_mask = torch.ones(self.num_outputs).to(self.device) * -10000000
        self.one_mask = torch.ones(self.num_outputs).to(self.device)

        lr_actor = learning_rate  # learning rate for actor network 0003
        lr_critic = learning_rate  # learning rate for critic network 001

        # Fully connected values:
        self.fc_dim = 136
        self.num_fc = 2

        self.actor = nn.Sequential(
            nn.Linear(
                get_flat_obs_size(obs_space.spaces["flat"]),
                self.num_outputs,
            ),
            nn.ReLU(),
        )

        self.fc_layers_val_layers = []

        for _ in range(self.num_fc):
            self.fc_layers_val_layers.append(nn.Linear(self.fc_dim, self.fc_dim))
            self.fc_layers_val_layers.append(nn.ReLU())

        self.fc_layers_val_layers.append(nn.Linear(self.fc_dim, 1))

        self.critic = nn.Sequential(*self.fc_layers_val_layers)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.actor.parameters(), "lr": lr_actor},
                {"params": self.critic.parameters(), "lr": lr_critic},
            ]
        )

    def act(self, obs):
        """
        Args:
            obs: agent environment observation

        Returns:
            action: taken action
            action_logprob: log probability of that action
        """
        obs2 = {}
        for key in obs.keys():
            obs2[key] = torch.from_numpy(obs[key]).to(self.device).detach()

        action_probs = self.actor(obs2["flat"])

        # Apply logits mask
        logit_mask = self.logit_mask * (
            self.one_mask - obs["action_mask"].reshape(self.num_outputs)
        )
        action_probs = action_probs + logit_mask

        dist = torch.distributions.Categorical(logits=action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach().numpy(), action_logprob.detach().numpy()

    def get_actions(self, obs: dict) -> dict:
        """
        Args:
            obs: agents environment observation

        Returns:
            action: taken action
        """
        actions = {a: None for a in obs.keys()}
        for agent in obs.keys():
            if agent != "p":
                actions[agent] = self.act(obs[agent])[0]

        return actions

    def evaluate(self, obs, act):
        """
        Args:
            obs: agent environment observation
            act: action that is mapped with

        Returns:
            action_logprobs: log probability that `act` is taken with this model
            state_values: value function reward prediction
            dist_entropy: entropy of actions distribution
        """
        action_probs = self.actor(obs["flat"])  # .squeeze())
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(act)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs["flat"])  # .squeeze())

        return action_logprobs.detach(), state_values.detach(), dist_entropy

    def forward(
        self,
    ):
        """
        Just don't.
        """
        raise NotImplementedError("Don't use this method.")

    def get_weights(self) -> dict:
        """
        Get policy weights.

        Return:
            actor_weights, critic_weights
        """
        actor_weights = self.actor.state_dict(keep_vars=False)

        critic_weights = self.critic.state_dict(keep_vars=False)

        optimizer_weights = 0
        return actor_weights, critic_weights, optimizer_weights

    def set_weights(self, actor_weights: dict, critic_weights: dict, optimizer_weights):
        """
        Set policy weights.

        Args:
            actor_weights: actor weights dictionary - from numpy
            critic_weights: critic weights dictionary - from numpy
        """
        self.actor.load_state_dict(actor_weights)
        self.critic.load_state_dict(critic_weights)


class PytorchLinearP(nn.Module):
    """A linear (feed-forward) model."""

    def __init__(self, obs_space, action_space, device, learning_rate):
        super().__init__()
        self.device = device

        self.mask_name = "action_mask"
        self.num_outputs = action_space
        self.logit_mask = torch.ones(self.num_outputs).to(self.device) * -10000000
        self.one_mask = torch.ones(self.num_outputs).to(self.device)

        lr_actor = learning_rate  # learning rate for actor network 0003
        lr_critic = learning_rate  # learning rate for critic network 001

        # Fully connected values:
        self.fc_dim = 86
        self.num_fc = 2

        self.actor = nn.Sequential(
            nn.Linear(
                get_flat_obs_size(obs_space.spaces["flat"]),
                self.num_outputs,
            ),
            nn.ReLU(),
        )

        self.fc_layers_val_layers = []

        for _ in range(self.num_fc):
            self.fc_layers_val_layers.append(nn.Linear(self.fc_dim, self.fc_dim))
            self.fc_layers_val_layers.append(nn.ReLU())

        self.fc_layers_val_layers.append(nn.Linear(self.fc_dim, 1))

        self.critic = nn.Sequential(*self.fc_layers_val_layers)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr_actor},
                {"params": self.critic.parameters(), "lr": lr_critic},
            ]
        )

    def act(self, obs):
        """
        Args:
            obs: agent environment observation

        Returns:
            action: taken action
            action_logprob: log probability of that action
        """
        obs2 = {}
        for key in obs.keys():
            obs2[key] = torch.from_numpy(obs[key]).to(self.device).detach()

        action_probs = self.actor(obs2["flat"])

        # Apply logits mask
        logit_mask = self.logit_mask * (
            self.one_mask - obs["action_mask"].reshape(self.num_outputs)
        )
        action_probs = action_probs + logit_mask

        action_probs = torch.split(action_probs, 22)

        action = torch.zeros(7)
        action_logprob = torch.zeros(7)

        for i, sub_prob in enumerate(action_probs):
            dist = torch.distributions.Categorical(logits=sub_prob)

            action[i] = dist.sample()
            action_logprob[i] = dist.log_prob(action[i])

        return action.detach().numpy(), action_logprob.detach().numpy()

    def evaluate(self, obs, act):
        """
        Args:
            obs: agent environment observation
            act: action that is mapped with

        Returns:
            action_logprobs: log probability that `act` is taken with this model
            state_values: value function reward prediction
            dist_entropy: entropy of actions distribution
        """
        
        action_probs = self.actor(obs["flat"]) 
        action_probs = torch.split(action_probs, 22, dim=-1)

        dist_entropy = torch.zeros((7, len(act)))
        action_logprobs = torch.zeros((7, len(act)))

        for i, sub_prob in enumerate(action_probs):
            dist = torch.distributions.Categorical(logits=sub_prob)

            action_logprobs[i] = dist.log_prob(act[:, i])
            dist_entropy[i] = dist.entropy()

        state_values = self.critic(obs["flat"]) 

        action_logprobs = action_logprobs.transpose(0, 1)
        dist_entropy = dist_entropy.transpose(0, 1)
        
        return action_logprobs.detach(), state_values.detach(), dist_entropy

    def forward(
        self,
    ):
        """
        Just don't.
        """
        raise NotImplementedError("Don't use this method.")

    def get_weights(self) -> dict:
        """
        Get policy weights.

        Return:
            actor_weights, critic_weights
        """
        actor_weights = self.actor.state_dict(keep_vars=False)

        critic_weights = self.critic.state_dict(keep_vars=False)

        optimizer_weights = 0
        return actor_weights, critic_weights, optimizer_weights

    def set_weights(self, actor_weights: dict, critic_weights: dict, optimizer_weights):
        """
        Set policy weights.

        Args:
            actor_weights: actor weights dictionary - from numpy
            critic_weights: critic weights dictionary - from numpy
        """
        self.actor.load_state_dict(actor_weights)
        self.critic.load_state_dict(critic_weights)
