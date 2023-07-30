"""
Rollout worker.Manages a policy and creates a batch.

Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""
import logging
import numpy as np
from typing import Tuple

from src.common import EmptyModel
from src.common.rollout_buffer import RolloutBuffer
from src.train.ppo.utils.execution_time import exec_time
from src.train.ppo import PpoPolicy
from src.train.ppo.utils import data_logging, save_batch

# pylint: disable=consider-using-dict-items,consider-iterating-dictionary


class RolloutWorker:
    """
    """

    def __init__(
        self,
        rollout_fragment_length: int,
        batch_iterations: int,
        policies_config: dict,
        mapping_function,
        actor_keys: list,
        env,
        seed: int,
        _id: int = -1,
        experiment_name=None,
    ):
        self.env = env
        self._id = _id

        self.actor_keys = actor_keys
        self.batch_iterations = batch_iterations
        self.rollout_fragment_length = rollout_fragment_length
        self.batch_size = self.batch_iterations * self.rollout_fragment_length
        self.policy_mapping_function = mapping_function
        self.experiment_name = experiment_name

        logging.debug(
            "ID: %s, Length: %s, iter:%s, batch_size:%s",
            self._id,
            self.rollout_fragment_length,
            self.batch_iterations,
            self.batch_size,
        )

        policy_keys = policies_config.keys()
        env.seed(seed + _id)
        env_keys = env.reset().keys()

        if self._id != -1:
            string = f"{','.join(map(str, env_keys))}\n"
            data_logging(data=string, experiment_id=self.experiment_name, id=self._id)
        else:
            string = (
                "a_actor_loss,a_critic_loss,a_entropy,p_a_loss,p_c_loss,p_entropy\n"
            )
            data_logging(data=string, experiment_id=self.experiment_name, id=self._id)

        # Build policices
        self.policies = {}
        for key in policy_keys:
            self.policies[key] = self._build_policy(policies_config[key])

        obs = env.reset()

        self.memory = RolloutBuffer(obs, self.policy_mapping_function)
        self.rolling_memory = RolloutBuffer(obs, self.policy_mapping_function)

        logging.debug("Rollout Worker %s built", self._id)

    def _build_policy(self, policy_config: dict):
        if policy_config["policy"] == EmptyModel:
            return EmptyModel(
                observation_space=policy_config["observation_space"],
                action_space=policy_config["action_space"],
            )
        elif policy_config["policy"] == PpoPolicy:
            return PpoPolicy(
                observation_space=policy_config["observation_space"],
                action_space=policy_config["action_space"],
                K_epochs=policy_config["k_epochs"],
                eps_clip=policy_config["eps_clip"],
                gamma=policy_config["gamma"],
                learning_rate=policy_config["learning_rate"],
                c1=policy_config["c1"],
                c2=policy_config["c2"],
                device=policy_config["device"],
                name=policy_config["name"],
            )

    # @exec_time
    def batch(self):
        """
        Creates a batch of `rollout_fragment_length` steps, save in `self.rollout_buffer`.
        """
        # reset batching environment and get its observation
        obs = self.env.reset()

        # reset rollout_buffer
        self.memory.clear()

        static_planner_action, static_planner_action_logprob = self.get_actions(obs)
        static_planner_action = static_planner_action["p"]
        static_planner_action_logprob = static_planner_action_logprob["p"]

        for i in range(self.batch_iterations):
            logging.debug(" ID: %s -- iteration: %s", self._id, i)

            for _ in range(self.rollout_fragment_length):
                # get actions, action_logprob for all agents in each policy* wrt observation
                policy_action, policy_logprob = self.get_actions(obs)

                # set static planner action
                policy_action["p"] = static_planner_action
                policy_logprob["p"] = static_planner_action_logprob

                # get new_observation, reward, done from stepping the environment
                next_obs, rew, done, _ = self.env.step(policy_action)

                if done["__all__"] is True:
                    next_obs = self.env.reset()

                self.rolling_memory.update(
                    action=policy_action,
                    logprob=policy_logprob,
                    state=obs,
                    reward=rew,
                    is_terminal=done["__all__"],
                )

                obs = next_obs

            self.memory.extend(self.rolling_memory)
            self.rolling_memory.clear()

        # Dump memory in ram
        save_batch(data=self.memory, worker_id=self._id)

    def get_actions(self, obs: dict) -> Tuple[dict, dict]:
        """
        Build action dictionary using actions taken from all policies.

        Args:
            obs: environment observation

        Returns:
            policy_action
            policy_logprob
        """

        policy_action, policy_logprob = {}, {}

        for key in obs.keys():
            (policy_action[key], policy_logprob[key]) = self.policies[
                self.policy_mapping_function(key)
            ].act(obs[key])

        return policy_action, policy_logprob

    def learn(self, memory) -> float:
        """
        Call the learning function for each policy.
        """
        losses = []
        data = {key : {} for key in self.policies}
        for key in self.policies:
            logging.debug("learning actor: %s", key)
            tmp_data = self.policies[key].learn(rollout_buffer=memory[key])
            losses.append((tmp_data.get("loss", 0.0), tmp_data.get("a_loss", 0.0), tmp_data.get("c_loss", 0.0), tmp_data.get("entropy", 0.0)))

            data[key] = tmp_data

        rewards = []
        for _m in losses:
            for _k in _m:
                rewards.append(_k)

        _data = f"{','.join(map(str, rewards))}\n"

        data_logging(data=_data, experiment_id=self.experiment_name, id=self._id)

        return data

    def log_rewards(self):
        """
        Append agent's total reward for this batch
        """
        data = []
        data.append((self.memory.buffers["0"].rewards.sum()).item())
        data.append((self.memory.buffers["1"].rewards.sum()).item())
        data.append((self.memory.buffers["2"].rewards.sum()).item())
        data.append((self.memory.buffers["3"].rewards.sum()).item())
        data.append((self.memory.buffers["p"].rewards.sum()).item())
        
        data1 = f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]}\n"
        # with open(f"{self.logdir}/rewards/ppo.csv", "a+") as file:
        #         file.write(data)
        data_logging(data=data1, experiment_id=self.experiment_name, id=self._id)

    def get_weights(self) -> dict:
        """
        Get model weights
        """
        weights = {}
        for key in self.policies.keys():
            weights[key] = self.policies[key].get_weights()

        return weights

    def set_weights(self, weights: dict):
        """
        Set model weights
        """
        for key in self.policies.keys():
            self.policies[key].set_weights(weights[key])

    def save_models(self, name: str = None):
        """
        Save the model of each policy.
        """
        for key in self.policies.keys():
            if name is None:
                self.policies[key].save_model(
                    "experiments/" + self.experiment_name + f"/models/{key}.pt"
                )
            else:
                self.policies[key].save_model(
                    "experiments/" + self.experiment_name + f"/models/{key}_{name}.pt"
                )

    def load_models(self, models_to_load: dict, name: str = None):
        """
        Load the model of each policy.

        It doesn't load 'p' policy.
        """
        for key in models_to_load.keys():
            if name is None:
                self.policies[key].load_model(
                    "experiments/" + models_to_load[key] + f"/models/{key}.pt"
                )
            else:
                self.policies[key].load_model(
                    "experiments/" + models_to_load[key] + f"/models/{key}_{name}.pt"
                )
