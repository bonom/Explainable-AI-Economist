"""
Buffer used to store batched data

Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""
import logging
import numpy as np
import torch
from tensordict import TensorDict


class RolloutBuffer:
    """
    MultiAgent Rollout Buffer

    Args:
        environment observation -> unmapped.

    """

    def __init__(self, state: dict, mapping_function):
        self.actors_keys = [str(key) for key in state.keys()]

        self.buffers = {}
        for key in self.actors_keys:
            self.buffers[key]: SingleBuffer = SingleBuffer(state[key])

    def update(
        self, action: dict, logprob: dict, state: dict, reward: dict, is_terminal: bool
    ):
        """
        Append single iteration to the buffer.
        """
        for key in self.actors_keys:
            self.buffers[key].update(
                action[key], logprob[key], state[key], reward[key], is_terminal
            )

    def extend(self, other):
        """
        Concatenate current `RolloutBuffer` with anotherone with the same structure.
        Note: there are no checks on this action.
        """
        for key in self.actors_keys:
            self.buffers[key].extend(other.buffers[key])

    def to_tensor(self):
        """
        Could be done better.
        """
        tensored_buffer = {}
        self.buffers["0"].extend(self.buffers["1"])
        self.buffers["0"].extend(self.buffers["2"])
        self.buffers["0"].extend(self.buffers["3"])

        tensored_buffer["a"] = self.buffers["0"].to_tensor()
        if "p" in self.buffers.keys():
            tensored_buffer["p"] = self.buffers["p"].to_tensor()

        return tensored_buffer

    def clear(self):
        for key in self.actors_keys:
            self.buffers[key].clear()


class SingleBuffer:
    def __init__(self, state: dict):
        self.empty = True

        self.actions = None
        self.states = {}
        self.logprobs = None
        self.rewards = None
        self.is_terminals = None

        self.state_keys_dict = {}
        for key in state.keys():
            self.states[key] = None
            self.state_keys_dict[key] = None

    def __append_tensor(self, old_stack, new_tensor):
        """
        Appends two tensors on a new axis. If `old_stack` shape is the same of `new_tensor`
        they are stacked together with `np.stack`. Otherwise if `old_stack` shape is bigger
        than `new_tensor` shape's, `new_tensor` is expanded on axis 0 and they are concatenated.

        Args:
            old_stack: old stack, it's shape must be >= new_tensor's shape
            new_tensor: new tensor that will be appended to `old_stack`
        """
        shape = (1, 1)
        if new_tensor.shape:
            shape = (1, *new_tensor.shape)

        nt = np.reshape(new_tensor, shape)

        if old_stack is None or old_stack is {}:
            return nt
        else:
            return np.concatenate((old_stack, nt))

    def update(self, action, logprob, state, reward, is_terminal):
        self.actions = self.__append_tensor(self.actions, action)
        self.logprobs = self.__append_tensor(self.logprobs, logprob)
        self.rewards = self.__append_tensor(self.rewards, reward)
        self.is_terminals = self.__append_tensor(self.is_terminals, is_terminal)

        for key in self.states.keys():
            self.states[key] = self.__append_tensor(self.states[key], state[key])
        self.empty = False

    def extend(self, other):
        
        if self.empty is True:
            self.actions = other.actions
            self.logprobs = other.logprobs
            self.rewards = other.rewards
            self.is_terminals = other.is_terminals

            for key in self.states.keys():
                self.states[key] = other.states[key]
        else:
            self.actions = np.concatenate((self.actions, other.actions))
            self.logprobs = np.concatenate((self.logprobs, other.logprobs))

            self.rewards = np.concatenate((self.rewards, other.rewards))
            self.is_terminals = np.concatenate((self.is_terminals, other.is_terminals))

            for key in self.states.keys():
                self.states[key] = np.concatenate((self.states[key], other.states[key]))

        self.empty = False

    def to_tensor(self):
        _buffer = SingleBuffer(self.state_keys_dict)

        _buffer.actions = torch.from_numpy(self.actions)
        _buffer.logprobs = torch.from_numpy(self.logprobs)
        _buffer.rewards = torch.from_numpy(self.rewards)
        _buffer.is_terminals = torch.from_numpy(self.is_terminals)

        _buffer.states = TensorDict(self.states, batch_size=[len(self.actions)])

        return _buffer

    def clear(self):
        self.actions = None
        self.logprobs = None
        self.rewards = None
        self.is_terminals = None

        for key in self.states.keys():
            self.states[key] = None

        self.empty = True
