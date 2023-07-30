"""
AI-Economist inspired pytorch model

Author: bonom
Github: https://github.com/bonom

Adapted from: Sa1g (https://github.com/sa1g)

Date: 2023-07-30

"""

import torch
import numpy as np
import torch.nn as nn

# from torch.autograd import Variable

WORLD_MAP = "world-map"
WORLD_IDX_MAP = "world-idx_map"
ACTION_MASK = "action_mask"

N_DIMS = {
    "world-map": 4,
    "world-idx_map": 4,
    "flat": 2,
    "time": 2,
    "action_mask": 2,
    "p0": 2,
    "p1": 2,
    "p2": 2,
    "p3": 2,
}


class LSTMModel(nn.Module):
    """
    Actor&Critic (Policy) Model.
    """

    def __init__(
            self, 
            obs_space: dict, 
            action_space: dict, 
            device: str = "cpu"
        ):
        """
        Initialize the ActorCritic Model.
        """
        super().__init__()

        self.action_space = action_space
        self.device = device

        ### Initialize some variables needed here
        self.cell_size = 128
        self.input_emb_vocab = 100
        self.emb_dim = 4
        self.num_conv = 2
        self.fc_dim = 128
        self.num_fc = 2
        self.filter = (16, 32)
        self.kernel_size = (3, 3)
        self.strides = 2
        self.lr = 0.0003
        self.momentum = 0.9

        if self.action_space == 22:
            self.output_dim = 7
        else:
            self.output_dim = 1

        ### Manage observation shape(s)
        self.conv_map_channels, self.conv_shape_r, self.conv_shape_c = obs_space.spaces[
            WORLD_MAP
        ].shape

        self.conv_idx_channels = obs_space.spaces[WORLD_IDX_MAP].shape[0] * self.emb_dim

        ### Embedding layers

        self.embed_map_idx_policy = nn.Embedding(
            self.input_emb_vocab, self.emb_dim, device=device, dtype=torch.float32
        )
        self.embed_map_idx_value = nn.Embedding(
            self.input_emb_vocab, self.emb_dim, device=device, dtype=torch.float32
        )
        self.conv_layers_policy = nn.ModuleList()
        self.conv_layers_value = nn.ModuleList()
        self.conv_shape = (
            self.conv_shape_r,
            self.conv_shape_c,
            self.conv_map_channels + self.conv_idx_channels,
        )

        for i in range(1, self.num_conv):
            if i == 1:
                self.conv_layers_policy.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=self.filter[0],
                        kernel_size=self.kernel_size,
                        stride=self.strides,
                    )
                )
                self.conv_layers_value.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=self.filter[0],
                        kernel_size=self.kernel_size,
                        stride=self.strides,
                    )
                )
            self.conv_layers_policy.append(
                nn.Conv2d(
                    in_channels=self.filter[0],
                    out_channels=self.filter[1],
                    kernel_size=self.kernel_size,
                    stride=self.strides,
                )
            )
            self.conv_layers_value.append(
                nn.Conv2d(
                    in_channels=self.filter[0],
                    out_channels=self.filter[1],
                    kernel_size=self.kernel_size,
                    stride=self.strides,
                )
            )

        self.conv_dims = 192 if self.action_space == 50 else 320
        self.flatten_dims = (
            self.conv_dims
            + obs_space.spaces["flat"].shape[0]
            + obs_space.spaces["time"].shape[0]
        )

        if self.action_space == 22:
            for agent in ["p0", "p1", "p2", "p3"]:  # 32
                self.flatten_dims += obs_space.spaces[agent].shape[0]

        self.fc_layer_1_policy = nn.Linear(
            in_features=self.flatten_dims, out_features=self.fc_dim
        )
        self.fc_layer_2_policy = nn.Linear(
            in_features=self.fc_dim, out_features=self.fc_dim
        )
        self.fc_layer_1_value = nn.Linear(
            in_features=self.flatten_dims, out_features=self.fc_dim
        )
        self.fc_layer_2_value = nn.Linear(
            in_features=self.fc_dim, out_features=self.fc_dim
        )
        self.lstm_policy = nn.LSTM(
            input_size=self.fc_dim,
            hidden_size=self.cell_size,
            num_layers=1,
        )
        self.lstm_value = nn.LSTM(
            input_size=self.fc_dim,
            hidden_size=self.cell_size,
            num_layers=1,
        )
        self.layer_norm_policy = nn.LayerNorm(self.fc_dim)
        self.layer_norm_value = nn.LayerNorm(self.fc_dim)
        self.output_policy = nn.Linear(
            in_features=self.cell_size, out_features=self.action_space
        )
        self.output_value = nn.Linear(in_features=self.cell_size, out_features=1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        # self.hidden_state_h_p = torch.ones(1, self.cell_size, device=self.device)
        # self.hidden_state_c_p = torch.ones(1, self.cell_size, device=self.device)
        # self.hidden_state_h_v = torch.ones(1, self.cell_size, device=self.device)
        # self.hidden_state_c_v = torch.ones(1, self.cell_size, device=self.device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.99
        )
        self.mse_loss = nn.MSELoss()

        # TODO: spostare in alto
        self.logit_mask = torch.ones(self.action_space).to(self.device) * -10000000
        self.one_mask = torch.ones(self.action_space).to(self.device)

    def apply_logit_mask(self, logits, mask):
        """
        Apply mask to logits, gets an action and calculates its log_probability.
        Mask values of 1 are valid actions.

        Args:
            obs: agent environment observation

        Returns:
            action: taken action
            action_logprob: log probability of that action
        """
        # Add huge negative values to logits with 0 mask values.
        logit_mask = self.logit_mask * (self.one_mask - mask)
        action_probs = logits + logit_mask

        dist = torch.distributions.Categorical(logits=action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob

    def __forward(self, obs):
        """
        Calculates action probability from the model.

        Args:
            obs: environment observation

        Returns:
            probability: ...
        """

        _world_map = obs[WORLD_MAP].int()
        _world_idx_map = obs[WORLD_IDX_MAP].int()
        _flat = obs["flat"]
        _time = obs["time"].int()
        # _action_mask = obs[ACTION_MASK].int()

        if self.action_space == 22:
            _p0 = obs["p0"]
            _p1 = obs["p1"]
            _p2 = obs["p2"]
            _p3 = obs["p3"]

        conv_input_map = torch.permute(_world_map, (0, 2, 3, 1))
        conv_input_idx = torch.permute(_world_idx_map, (0, 2, 3, 1))

        # Concatenate the remainings of the input
        if self.action_space == 22:
            non_convolutional_input = torch.cat(
                [
                    _flat,
                    _time,
                    _p0,
                    _p1,
                    _p2,
                    _p3,
                ],
                axis=1,
            )
        else:
            non_convolutional_input = torch.cat(
                [
                    _flat,
                    _time,
                ],
                axis=1,
            )

        # Policy
        # Embedd from 100 to 4
        map_embedd = self.embed_map_idx_policy(conv_input_idx)
        # Reshape the map
        map_embedd = torch.reshape(
            map_embedd,
            (-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels),
        )
        # Concatenate the map and the idx map
        conv_input = torch.cat([conv_input_map, map_embedd], axis=-1)

        # Convolutional Layers
        for conv_layer in self.conv_layers_policy:
            conv_output = conv_layer(conv_input)
            conv_input = self.relu(conv_output)

        # Flatten the output of the convolutional layers
        flatten = torch.reshape(
            conv_input, (-1, self.conv_dims)
        )  # 192 is from 32 * 3 * 2

        # Concatenate the convolutional output with the non convolutional input
        fc_in = torch.cat([flatten, non_convolutional_input], axis=-1)

        # Fully Connected Layers
        for i in range(self.num_fc):
            if i == 0:
                fc_in = self.relu(self.fc_layer_1_policy(fc_in))
            else:
                fc_in = self.relu(self.fc_layer_2_policy(fc_in))

        # Normalize the output
        layer_norm_out = self.layer_norm_policy(fc_in)
        # LSTM

        # Project LSTM output to logits
        # TODO:
        # lstm_out, hidden = self.lstm_policy(layer_norm_out, (self.hidden_state_h_p, self.hidden_state_c_p))
        # self.hidden_state_h_p, self.hidden_state_c_p = hidden[0].detach(), hidden[1].detach()
        lstm_out, _ = self.lstm_policy(layer_norm_out)
        logits = self.output_policy(lstm_out)

        return logits

    def act(self, obs):
        """
        Args:
            obs: agent environment observation

        Returns:
            action: taken action
            action_logprob: log probability of that action
        """
        for key in obs.keys():
            if isinstance(obs[key], np.ndarray):
                obs[key]: torch.Tensor = torch.from_numpy(obs[key])
            obs[key]: torch.Tensor = obs[key].to(self.device)
        
        if obs.get('world-map').dim() < 4:
            obs['world-map'] = obs['world-map'].unsqueeze(0)
        if obs.get('world-idx_map').dim() < 4:
            obs['world-idx_map'] = obs['world-idx_map'].unsqueeze(0)
        if obs.get('time').dim() < 2:
            obs['time'] = obs['time'].unsqueeze(0)
        if obs.get('flat').dim() < 2:
            obs['flat'] = obs['flat'].unsqueeze(0)
        if obs.get('action_mask').dim() < 2:
            obs['action_mask'] = obs['action_mask'].unsqueeze(0)

        logits = self.__forward(obs)

        _action_mask = obs[ACTION_MASK].int()

        # Mask the logits
        if self.action_space == 22:
            logits = torch.split(logits, self.action_space, dim=1)
            mask = torch.split(_action_mask, self.action_space, dim=1)

            # for batch, log in enumerate(logits):
            action = torch.zeros((self.output_dim, self.action_space))
            new_action_logprob = torch.zeros((self.output_dim, self.action_space))
            for idx, l in enumerate(logits):
                out_action, action_logprob = self.apply_logit_mask(l, mask[idx])
                action[idx] = out_action
                new_action_logprob[idx] = action_logprob

            action_logprob = new_action_logprob
        else:
            action, action_logprob = self.apply_logit_mask(logits, _action_mask)

        return action, action_logprob

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
        for key in obs.keys():
            # if key in ['world-map', 'world-idx_map']:
            if key == "time":
                if len(obs[key].shape) < 2:
                    obs[key] = obs[key].unsqueeze(-1)
                elif len(obs[key].shape) > 2:
                    obs[key] = obs[key].squeeze(1)

                obs[key] = obs[key].to(self.device).int()
            else:
                obs[key] = obs[key].squeeze(1).to(self.device)
            # else:
            #     obs[key] = obs[key].to(self.device).squeeze(-1)

        action_probs = self.__forward(obs)
        dist = torch.distributions.Categorical(logits=action_probs)

        action_logprobs = dist.log_prob(act)
        dist_entropy = dist.entropy()

        _world_map = obs[WORLD_MAP].int()
        _world_idx_map = obs[WORLD_IDX_MAP].int()
        _flat = obs["flat"]
        _time = obs["time"].int()

        if self.action_space == 22:
            _p0 = obs["p0"]
            _p1 = obs["p1"]
            _p2 = obs["p2"]
            _p3 = obs["p3"]

        conv_input_map = torch.permute(_world_map, (0, 2, 3, 1))
        conv_input_idx = torch.permute(_world_idx_map, (0, 2, 3, 1))
        # # CRITIC
        # Embedd from 100 to 4
        map_embedd = self.embed_map_idx_value(conv_input_idx)

        # Reshape the map
        map_embedd = torch.reshape(
            map_embedd,
            (-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels),
        )

        if self.action_space == 22:
            non_convolutional_input = torch.cat(
                [
                    _flat,
                    _time,
                    _p0,
                    _p1,
                    _p2,
                    _p3,
                ],
                axis=1,
            )
        else:
            non_convolutional_input = torch.cat(
                [
                    _flat,
                    _time,
                ],
                axis=1,
            )

        # Concatenate the map and the idx map
        conv_input = torch.cat([conv_input_map, map_embedd], axis=-1)

        # Convolutional Layers
        for conv_layer in self.conv_layers_value:
            conv_input = self.relu(conv_layer(conv_input))

        # Flatten the output of the convolutional layers
        flatten = torch.reshape(
            conv_input, (-1, self.conv_dims)
        )  # 192 is from 32 * 3 * 2

        # Concatenate the convolutional output with the non convolutional input
        fc_in = torch.cat([flatten, non_convolutional_input], axis=-1)

        # Fully Connected Layers
        for i in range(self.num_fc):
            if i == 0:
                fc_in = self.relu(self.fc_layer_1_value(fc_in))
            else:
                fc_in = self.relu(self.fc_layer_2_value(fc_in))

        # Normalize the output
        layer_norm_out = self.layer_norm_value(fc_in)

        # Project LSTM output to logits
        # lstm_out, hidden = self.lstm_value(layer_norm_out, (self.hidden_state_h_p, self.hidden_state_c_p))
        # self.hidden_state_h_p, self.hidden_state_c_p = hidden[0].detach(), hidden[1].detach()

        lstm_out, _ = self.lstm_value(layer_norm_out)
        state_values = self.output_value(lstm_out)

        return action_logprobs, state_values, dist_entropy

    def get_weights(self) -> dict:
        """
        Get policy weights.

        Return:
            actor_weights, critic_weights
        """
        actor_weights = self.state_dict(keep_vars=True)
        critic_weights = 0
        optimizer_weights = 0

        return actor_weights, critic_weights, optimizer_weights

    def set_weights(self, actor_weights: dict, critic_weights: dict, optimizer_weights):
        """
        Set policy weights.

        Args:
            actor_weights: actor weights dictionary - from numpy
            critic_weights: critic weights dictionary - from numpy
        """
        self.load_state_dict(actor_weights)
