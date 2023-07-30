"""
Defines the configuration for training with online learning for DT.

Author(s): bonom, Sa1g
Github(s): https://github.com/bonom, https://github.com/sa1g

Date: 2023.07.30

"""
import os
import toml
import string
import random
import numpy as np


from datetime import datetime

from tqdm import tqdm
from deap import base, creator, tools
from joblib import Parallel, delayed, parallel_backend

from src.common import get_environment
from typing import Dict, List, Tuple, Union
from src.train.dt import PythonDT
from src.common.env import EnvWrapper
from ai_economist.foundation.base.base_env import BaseEnvironment
from src.train.dt import (
    GrammaticalEvolutionTranslator,
    grammatical_evolution,
    eaSimple,
)


class DtTrainConfig:
    """
    Endpoint to setup DT_ql's training configuration.

    Args:
        env: EnvWrapper
            The environment to train on. Should be the AI-Economist environment.
        seed: int
            The seed to use for reproducibility.
        lr: Union[float, str]
            The learning rate to use. If "auto", the learning rate will be automatically determined.
        df: float
            The discount factor to use.
        eps: float
            The epsilon value to use for epsilon-greedy.
        low: float
            The lower bound for the action space.
        up: float
            The upper bound for the action space.
        input_space: int
            The number of inputs to the decision tree.
        episodes: int
            The number of episodes to train for.
        episode_len: int
            The number of steps per episode.
        lambda_: int
            The lambda value to use for the evolution.
        generations: int
            The number of generations to train for.
        cxp: float
            The crossover probability to use.
        mp: float
            The mutation probability to use.
        mutation: Dict[str, Union[str, int, float]]
            The mutation function to use.
        crossover: Dict[str, Union[str, int, float]]
            The crossover function to use.
        selection: Dict[str, Union[str, int, float]]
            The selection function to use.
        genotype_len: int
            The length of the genotype.
        types: List[Tuple[int, int, int, int]]
            The types to use for the decision tree.
    """

    def __init__(
        self,
        env: EnvWrapper = None,
        agent: Union[bool, str] = True,
        planner: Union[bool, str] = True,
        seed: int = 1,
        lr: Union[float, str] = "auto",
        df: float = 0.9,
        eps: float = 0.05,
        low: float = -10,
        up: float = 10,
        input_space: int = 1,
        episodes: int = 500,
        episode_len: int = 1000,
        lambda_: int = 1000,
        generations: int = 1000,
        cxp: float = 0.5,
        mp: float = 0.5,
        mutation: Dict[str, Union[str, int, float]] = {
            "function": "tools.mutUniformInt",
            "low": 0,
            "up": 40000,
            "indpb": 0.1,
        },
        crossover: Dict[str, Union[str, int, float]] = {"function": "tools.cxOnePoint"},
        selection: Dict[str, Union[str, int, float]] = {
            "function": "tools.selTournament",
            "tournsize": 2,
        },
        genotype_len: int = 100,
        types: List[Tuple[int, int, int, int]] = None,
        mapped_agents: Dict[str, Union[bool, str]] = {
            "a": True,
            "p": True,
        },
    ):
        if env is not None:
            self.env = env
            self.agent = mapped_agents["a"]
            self.planner = mapped_agents["p"]

            # Set seeds
            assert seed >= 1, "Seed must be greater than 0"
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed = seed
            self.seed = seed

            # Add important variables to self
            self.episodes = episodes
            self.episode_len = episode_len
            # For the leaves
            self.n_actions = {
                "a": env.action_space.n,
                "p": env.action_space_pl.nvec[0].item(),
            }
            # For the input space
            obs = env.reset()
            input_space = {
                "a": obs.get("0").get("flat").shape[0],
                "p": obs.get("p").get("flat").shape[0],
            }
            self.lr = lr
            self.df = df
            self.eps = eps
            self.low = low
            self.up = up

            # For the evolution
            self.lambda_ = lambda_
            self.generations = generations
            self.cxp = cxp
            self.mp = mp
            self.genotype_len = genotype_len
            self.types = types

            # Create the log directory
            phase = "P1" if agent and not planner else "P2"
            date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.logdir = "experiments/DT_{}_{}_{}".format(phase, date, episodes)
            self.logfile = os.path.join(self.logdir, "log.txt")
            os.makedirs(self.logdir)

            # Convert the string to dict
            self.mutation = mutation
            self.crossover = crossover
            self.selection = selection

            # Initialize some variables
            grammar_agent = {
                "bt": ["<if>"],
                "if": ["if <condition>:{<action>}else:{<action>}"],
                "condition": [
                    "_in_{0}<comp_op><const_type_{0}>".format(k)
                    for k in range(input_space.get("a", 136))
                ],
                "action": ['out=_leaf;leaf="_leaf"', "<if>"],
                "comp_op": [" < ", " > "],
            }
            types_agent: str = (
                types
                if types is not None
                else ";".join(["0,10,1,10" for _ in range(input_space.get("a", 136))])
            )
            types_agent: str = types_agent.replace("#", "")
            assert len(types_agent.split(";")) == input_space.get(
                "a", 136
            ), "Expected {} types_agent, got {}.".format(
                input_space.get("a", 136), len(types_agent.split(";"))
            )

            for index, type_ in enumerate(types_agent.split(";")):
                rng = type_.split(",")
                start, stop, step, divisor = map(int, rng)
                consts_ = list(
                    map(str, [float(c) / divisor for c in range(start, stop, step)])
                )
                grammar_agent["const_type_{}".format(index)] = consts_

            # Add to self
            self.grammar_agent = grammar_agent

            grammar_planner = {
                "bt": ["<if>"],
                "if": ["if <condition>:{<action>}else:{<action>}"],
                "condition": [
                    "_in_{0}<comp_op><const_type_{0}>".format(k)
                    for k in range(input_space.get("p", 86))
                ],
                "action": ['out=_leaf;leaf="_leaf"', "<if>"],
                "comp_op": [" < ", " > "],
            }
            types_planner: str = (
                types
                if types is not None
                else ";".join(["0,10,1,10" for _ in range(input_space.get("p", 86))])
            )
            types_planner: str = types_planner.replace("#", "")
            assert len(types_planner.split(";")) == input_space.get(
                "p", 86
            ), "Expected {} types_planner, got {}.".format(
                input_space.get("p", 86), len(types_planner.split(";"))
            )

            for index, type_ in enumerate(types_planner.split(";")):
                rng = type_.split(",")
                start, stop, step, divisor = map(int, rng)
                consts_ = list(
                    map(str, [float(c) / divisor for c in range(start, stop, step)])
                )
                grammar_planner["const_type_{}".format(index)] = consts_

            # Add to self
            self.grammar_planner = grammar_planner

            # Log the configuration
            with open(os.path.join(self.logdir, "config.toml"), "w") as f:
                config_dict = {
                    "common": {
                        "algorithm_name": "DT",
                        "phase": 1 if not planner else 2,
                        "step": self.episode_len,
                        "seed": seed,
                        "device": "cpu",
                        "mapped_agents": mapped_agents,
                    },
                    "algorithm_specific": {
                        "episodes": episodes,
                        "generations": generations,
                        "cxp": cxp,
                        "mp": mp,
                        "genotype_len": genotype_len,
                        "types": types,
                        "mutation": self.mutation,
                        "crossover": self.crossover,
                        "selection": self.selection,
                        "lr": lr,
                        "df": df,
                        "eps": eps,
                        "low": low,
                        "up": up,
                        "lambda_": lambda_,
                        "input_space": input_space,
                        "grammar_agent": self.grammar_agent,
                        "grammar_planner": self.grammar_planner,
                    },
                }
                toml.dump(config_dict, f)

            # Check the variables
            self.__check_variables()

    def __check_variables(self):
        """
        Checks if all variables are set.
        """
        assert self.agent == True or (
            isinstance(self.agent, str)
            and os.path.exists(os.path.join("experiments", self.agent))
        ), "The agent must be trained or loaded from existing directory, received {}".format(
            self.agent
        )
        assert self.lr == "auto" or (
            isinstance(self.lr, float) and self.lr > 0 and self.lr < 1
        ), "{} is not known or not in the right range ({})".format(
            type(self.lr), self.lr
        )
        assert self.df > 0 and self.df < 1, "df must be between 0 and 1"
        assert self.eps > 0 and self.eps < 1, "eps must be between 0 and 1"
        assert self.low < self.up, "low must be smaller than up"
        assert self.episodes > 0, "episodes must be greater than 0"
        assert self.episode_len > 0, "episode_len must be greater than 0"
        assert self.lambda_ > 0, "lambda must be greater than 0"
        assert self.generations > 0, "generations must be greater than 0"
        assert (
            self.cxp > 0 and self.cxp < 1
        ), "Crossover probability must be between 0 and 1"
        assert (
            self.mp > 0 and self.mp < 1
        ), "Mutation probability must be between 0 and 1"
        assert self.genotype_len > 0, "Genotype length must be greater than 0"

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir, exist_ok=True)

        models_dir = os.path.join(self.logdir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)

        rewards_dir = os.path.join(self.logdir, "rewards")
        if not os.path.exists(rewards_dir):
            os.makedirs(rewards_dir, exist_ok=True)

        rewards_csv_file = os.path.join(rewards_dir, "dt.csv")
        with open(rewards_csv_file, "w") as f:
            f.write("0,1,2,3,p,fitnesses\n")

    def evaluate_fitness(
        self,
        genotype: List[int],
    ) -> float:
        # Get the phenotype
        phenotype_agent, _ = GrammaticalEvolutionTranslator(
            self.grammar_agent
        ).genotype_to_str(genotype)
        phenotype_planner, _ = GrammaticalEvolutionTranslator(
            self.grammar_planner
        ).genotype_to_str(genotype)

        # Get the Decision Trees of agents and planner
        dt = PythonDT(
            phenotype_agent,
            self.n_actions.get("a", 50),
            self.lr,
            self.df,
            self.eps,
            self.low,
            self.up,
        )
        dt_p = PythonDT(
            phenotype_planner,
            self.n_actions.get("p", 22),
            self.lr,
            self.df,
            self.eps,
            self.low,
            self.up,
            True,
        )
        if isinstance(self.agent, str) and os.path.exists(
            os.path.join("experiments", self.agent)
        ):
            dt.load(os.path.join("experiments", self.agent))

        if isinstance(self.planner, str) and os.path.exists(
            os.path.join("experiments", self.planner)
        ):
            dt_p.load(os.path.join("experiments", self.planner), planner=True)
        elif self.planner == False:
            dt_p = None

        # Evaluate the fitness
        return self.fitness(dt, dt_p)

    def stepper(
        self, agent_path: str, planner_path: str = None, env: BaseEnvironment = None
    ):
        """
        Stepper used for the `interact.py`
        """
        agent_path = os.path.join(agent_path, "models", "dt_a.pkl")
        planner_path = os.path.join(planner_path, "models", "dt_p.pkl")

        agent = PythonDT(load_path=agent_path)
        planner = (
            PythonDT(load_path=planner_path, planner=True)
            if planner_path is not None
            else None
        )

        # Initialize some variables
        obs: Dict[str, Dict[str, np.ndarray]] = env.reset(force_dense_logging=True)

        # Start the episode
        agent.new_episode()
        if planner is not None:
            planner.new_episode()

        # Run the episode
        for t in tqdm(range(env.env.episode_length), desc="DecisionTree Replay"):
            actions = agent.get_actions(obs)
            if planner is not None:
                actions["p"] = planner(obs.get("p").get("flat"))

            obs, rew, done, _ = env.step(actions)

            # Add reward to list
            agent.add_rewards(rewards=rew)
            if planner is not None:
                planner.add_rewards(rewards=rew)

            agent.set_reward(sum([rew.get(k, 0) for k in rew.keys() if k != "p"]))
            if planner is not None:
                planner.set_reward(rew.get("p", 0))

            if done["__all__"] is True:
                break

        return agent.rewards, env.env.previous_episode_dense_log

    def fitness(self, agent: PythonDT, planner: PythonDT):
        global_cumulative_rewards = []

        # try:
        for iteration in range(self.episodes):
            # Set the seed and reset the environment
            self.seed += 1
            self.env.seed = self.seed
            obs: Dict[str, Dict[str, np.ndarray]] = self.env.reset()

            # Start the episode
            agent.new_episode()
            if planner is not None:
                planner.new_episode()

            # Initialize some variables
            cum_global_rew = 0
            cum_rew = {key: np.empty((0)) for key in obs.keys()}

            # Run the episode
            for t in range(self.episode_len):
                actions = agent.get_actions(obs)
                if planner is not None:
                    actions["p"] = planner(obs.get("p").get("flat"))

                if any([a is None for a in actions.values()]):
                    break

                obs, rew, done, _ = self.env.step(actions)

                for key in obs.keys():
                    cum_rew[key] = np.concatenate((cum_rew[key], rew.get(key, np.nan)))

                # self.env.render() # FIXME: This is not working, see if needed
                agent.set_reward(sum([rew.get(k, 0) for k in rew.keys() if k != "p"]))
                if planner is not None:
                    planner.set_reward(rew.get("p", 0))

                # cum_global_rew += rew
                cum_global_rew += sum([rew.get(k, 0) for k in rew.keys()])

                if done["__all__"].item() is True:
                    break

            new_rewards = {}
            for key in obs.keys():
                new_rewards[key] = np.sum(cum_rew.get(key, np.nan))

            agent.add_rewards(rewards=new_rewards)
            if planner is not None:
                planner.add_rewards(rewards=new_rewards)

            global_cumulative_rewards.append(cum_global_rew)

        fitness = (np.mean(global_cumulative_rewards),)
        return fitness, agent, planner

    def train(
        self,
    ) -> None:
        with parallel_backend("multiprocessing"):
            pop, log, hof, best_leaves = grammatical_evolution(
                self.evaluate_fitness,
                individuals=self.lambda_,
                generations=self.generations,
                cx_prob=self.cxp,
                m_prob=self.mp,
                logfile=self.logfile,
                mutation=self.mutation,
                crossover=self.crossover,
                initial_len=self.genotype_len,
                selection=self.selection,
            )
