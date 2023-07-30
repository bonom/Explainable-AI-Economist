"""
Implementation of the decision trees

Author: Leonardo Lucio Custode

Adapted from: bonom (https://github.com/bonom)
Date: 2023-07-30

"""
import os
import abc
import pickle
import numpy as np

from typing import List, Dict, Tuple, Union, Optional, Callable


class DecisionTree:
    def __init__(self):
        self.current_reward = 0
        self.last_leaf = None

    @abc.abstractmethod
    def get_action(self, input):
        pass

    def set_reward(self, reward):
        self.current_reward = reward

    def new_episode(self):
        self.last_leaf = None


class Leaf:
    def get_action(self):
        pass

    def update(self, x):
        pass


class QLearningLeaf(Leaf):
    def __init__(self, n_actions, learning_rate, discount_factor):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.parent = None
        self.iteration = [1] * n_actions

        self.q = np.zeros(n_actions, dtype=np.float32)
        self.last_action = 0

    def get_action(self):
        action = np.argmax(self.q)
        self.last_action = action
        return action

    def update(self, reward, qprime):
        if self.last_action is not None:
            lr = (
                self.learning_rate
                if not callable(self.learning_rate)
                else self.learning_rate(self.iteration[self.last_action])
            )
            if lr == "auto":
                lr = 1 / self.iteration[self.last_action]
            self.q[self.last_action] += lr * (
                reward + self.discount_factor * qprime - self.q[self.last_action]
            )

    def next_iteration(self):
        self.iteration[self.last_action] += 1

    def __repr__(self):
        return ", ".join(["{:.2f}".format(k) for k in self.q])

    def __str__(self):
        return repr(self)


class EpsGreedyLeaf(QLearningLeaf):
    def __init__(self, n_actions, learning_rate, discount_factor, epsilon):
        super().__init__(n_actions, learning_rate, discount_factor)
        self.epsilon = epsilon

    def get_action(self):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Get the argmax. If there are equal values, choose randomly between them
            best = []
            max_ = -float("inf")

            for i, v in enumerate(self.q):
                if v > max_:
                    max_ = v
                    best = [i]
                elif v == max_:
                    best.append(i)

            action = np.random.choice(best)

        self.last_action = action
        self.next_iteration()
        return action


class RandomlyInitializedEpsGreedyLeaf(EpsGreedyLeaf):
    def __init__(
        self, n_actions, learning_rate, discount_factor, epsilon, low=-100, up=100
    ):
        """
        Initialize the leaf.
        Params:
            - n_actions: The number of actions
            - learning_rate: the learning rate to use, callable or float
            - discount_factor: the discount factor, float
            - epsilon: epsilon parameter for the random choice
            - low: lower bound for the initialization
            - up: upper bound for the initialization
        """
        super(RandomlyInitializedEpsGreedyLeaf, self).__init__(
            n_actions, learning_rate, discount_factor, epsilon
        )
        self.q = np.random.uniform(low, up, n_actions)


class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    def __init__(
        self,
        n_actions: int,
        lr: Union[float, str],
        df: float,
        eps: float,
        low: float,
        up: float,
    ) -> None:
        super(CLeaf, self).__init__(n_actions, lr, df, eps, low=low, up=up)


class PythonDT(DecisionTree):
    def __init__(
        self,
        phenotype: str = None,
        n_actions: int = 0,
        lr: Union[float, str] = "auto",
        df: float = 0.9,
        eps: float = 0.05,
        low: float = -100,
        up: float = 100,
        # planner: bool = False,
        load_path: str = None,
    ) -> None:
        super(PythonDT, self).__init__()
        self.program = phenotype
        self.leaves = {}
        # self.planner = planner
        n_leaves = 0
        self.rewards = []

        if load_path is not None:
            self.load(load_path)

        else:
            while "_leaf" in self.program:
                new_leaf = CLeaf(n_actions, lr, df, eps, low=low, up=up)
                leaf_name = "leaf_{}".format(n_leaves)
                self.leaves[leaf_name] = new_leaf

                self.program = self.program.replace(
                    "_leaf", "'{}.get_action()'".format(leaf_name), 1
                )
                self.program = self.program.replace(
                    "_leaf", "{}".format(leaf_name), 1
                )

                n_leaves += 1

        self.exec_ = compile(self.program, "<string>", "exec", optimize=2)
        
    def get_action(self, input):
        if len(self.program) == 0:
            return None
        variables = {} 
        for idx, i in enumerate(input):
            variables["_in_{}".format(idx)] = i
        variables.update(self.leaves)

        exec(self.exec_, variables)

        current_leaf = self.leaves[variables["leaf"]]
        current_q_value = max(current_leaf.q)
        if self.last_leaf is not None:
            self.last_leaf.update(self.current_reward, current_q_value)
        self.last_leaf = current_leaf 
        
        return current_leaf.get_action()

    def __call__(self, x):
        return self.get_action(x)

    def __str__(self):
        return self.program

    def get_actions(self, inputs: Dict[str, Dict[str, np.ndarray]]):
        actions = {}
        for agent, x in inputs.items():
            if agent != "p":
                actions[agent] = self.get_action(x.get("flat"))

        return actions

    def save(self, save_path: str):
        data = {
            "leaves": self.leaves,
            "program": self.program,
        }

        # Save self as a pickle
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, load_path: str):#, planner: bool = False):
        # Load self from a pickle
        with open(
            load_path,
            "rb",
        ) as f:
            data = pickle.load(f)
            self.leaves = data.get("leaves")
            self.program = data.get("program")

            self.exec_ = compile(self.program, "<string>", "exec", optimize=2)


class ForestTree:
    def __init__(
        self,
        phenotypes: List[str] = None,
        n_actions: int = 0,
        lr: Union[float, str] = "auto",
        df: float = 0.9,
        eps: float = 0.05,
        low: float = -100,
        up: float = 100,
        num_actions: int = 7,
        load_path: str = None,
    ) -> None:

        if load_path is not None:
            if os.path.exists(load_path):
                self.trees = {
                    i: PythonDT(load_path=os.path.join(load_path, 'dt_{}.pkl'.format(i))) for i in range(num_actions)
                }
            else:
                raise ValueError(f"Path '{load_path}' does not exists!")

        else:
            if not phenotypes or any([p is None for p in phenotypes]):
                raise ValueError("Phenotype is None and load_path is None")
            assert n_actions > 0, "n_actions must be greater than 0"

            self.trees = {}
            for key in range(num_actions):
                self.trees[key] = PythonDT(phenotypes[key],n_actions,lr,df,eps,low,up) 
           
        self.rewards = []

    def __call__(self, input):
        return self.get_actions(input)

    def __str__(self):
        string = ""

        for key, tree in self.trees.items():
            string += "\n\t- DT {} - \n\n{}\n\n".format(key+1, tree.program)

        return string
    
    @property
    def leaves(self,):
        return {
            key: tree.leaves for key, tree in self.trees.items()
        }

    def new_episode(self,):
        for tree in self.trees.values():
            tree.new_episode()

    def clear_rewards(self):
        self.rewards = []

    def add_rewards(self, rewards: Dict[str, float]):
        self.rewards.append(rewards)

    def get_last_rewards(self):
        return self.rewards[-1]

    def get_rewards(self):
        try:
            ret = {
                agent: np.array([r[agent] for r in self.rewards]).mean()
                for agent in self.rewards[0].keys()
            }
        except IndexError:
            ret = {agent: -np.inf for agent in ['0', '1', '2', '3', 'p']}

        return ret

    def get_actions(self, input):
        
        actions = np.empty((0))
        
        for tree in self.trees.values():
            actions = np.concatenate((actions, np.array([tree.get_action(input)])))
        
        return actions

    def set_reward(self, reward):
        for tree in self.trees.values():
            tree.set_reward(reward)

    def save(self, save_path):
        for idx, tree in self.trees.items():
            tree.save(os.path.join(save_path, "dt_{}.pkl".format(idx)))