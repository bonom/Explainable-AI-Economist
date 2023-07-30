"""
Defines the configuration for training with online learning for both PPO and DT.

Author(s): bonom, Sa1g
Github(s): https://github.com/bonom, https://github.com/sa1g

Date: 2023.07.30
"""

import os
import toml
import torch
import random
import logging
import numpy as np

from datetime import datetime
from src.common.env import EnvWrapper
from deap import base, creator, tools
from typing import Dict, List, Tuple, Union
from src.common.rollout_buffer import RolloutBuffer
from src.train.dt import (
    varAnd,
    ForestTree,
    ListWithParents,
    GrammaticalEvolutionTranslator,
)
from src.train.ppo import PpoPolicy
from src.train.ppo.models.linear import PytorchLinearA


class PPODtTrainConfig:
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
        episodes: int = 1,
        episode_len: int = 1000,
        lambda_: int = 100,
        generations: int = 50,
        cxp: float = 0.8,
        mp: float = 0.8,
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
        batch_size: int = None,
        rollout_fragment_length: int = 200,
        k_epochs: int = 16,
        eps_clip: int = 10,
        gamma: float = 0.998,
        device: str = "cpu",
        learning_rate: float = 0.0003,
        _c1: float = 0.05,
        _c2: float = 0.025,
        mapped_agents: Dict[str, Union[bool, str]] = {
            "a": True,
            "p": True,
        },
    ):
        if env is not None:
            # Ordinare questo file:
            # TODO: salvare TUTTI i parametri nel file toml, non solo alcuni

            # Logica:
            # per esempio vogliamo che ogni DT si alleni per 6000 passi, che ci siano 30 DT e che questi

            self.env = env
            self.set_seed(seed)
            obs = env.reset()

            self.build_ppo(
                rollout_fragment_length,
                batch_size,
                episode_len,
                episodes,
                k_epochs,
                eps_clip,
                gamma,
                device,
                learning_rate,
                _c1,
                _c2,
                mapped_agents,
                obs,
            )

            self.build_dt(
                mapped_agents,
                episodes,
                episode_len,
                obs,
                lr,
                df,
                eps,
                low,
                up,
                lambda_,
                generations,
                cxp,
                mp,
                genotype_len,
                types,
                mutation,
                crossover,
                selection,
                input_space,
            )

            self.create_directories_and_log_files(agent, planner)

            self.log_config(planner, mapped_agents, input_space)

            self.validate_config_dt()
            self.validate_config_ppo()

            # logging.info("Setup complete!")

    def log_config(self, planner, mapped_agents, input_space):
        """
        Put experiment's config in a TOML file.
        """
        with open(os.path.join(self.logdir, "config.toml"), "w") as f:
            config_dict = {
                "common": {
                    "algorithm_name": "PPO_DT",
                    "phase": 1 if not planner else 2,
                    "step": self.episode_len,
                    "seed": self.seed,
                    "device": "cpu",
                    "mapped_agents": mapped_agents,
                    "fitness_method": "Planner rewards"
                },
                "algorithm_specific": {
                    "dt": {
                        "episodes": self.episodes,
                        "generations": self.generations,
                        "cxp": self.cxp,
                        "mp": self.mp,
                        "genotype_len": self.genotype_len,
                        "types": self.types,
                        "mutation": self.mutation,
                        "crossover": self.crossover,
                        "selection": self.selection,
                        "lr": self.lr,
                        "df": self.df,
                        "eps": self.eps,
                        "low": self.low,
                        "up": self.up,
                        "lambda_": self.lambda_,
                        "input_space": input_space,
                        # "grammar_agent": self.grammar_agent,
                        "grammar_planner": {
                            str(key) : value for key, value in self.grammar_planner.items()
                        }
                    },
                    "ppo": {
                        "rollout_fragment_length": self.rollout_fragment_length,
                        "batch_size": self.batch_size,
                        "k_epochs": self.k_epochs,
                        "eps_clip": self.eps_clip,
                        "gamma": self.gamma,
                        "learning_rate": self.learning_rate,
                        "c1": self._c1,
                        "c2": self._c2,
                    },
                },
            }
            toml.dump(config_dict, f)

    def build_dt(
        self,
        mapped_agents,
        episodes,
        episode_len,
        obs,
        lr,
        df,
        eps,
        low,
        up,
        lambda_,
        generations,
        cxp,
        mp,
        genotype_len,
        types,
        mutation,
        crossover,
        selection,
        input_space,
    ):
        """
        Setup specific parameters for the decision tree.
        """
        self.planner = mapped_agents["p"]

        # Add important variables to self
        self.episodes = episodes
        self.episode_len = episode_len

        # For the leaves
        self.n_actions = {
            "a": self.env.action_space.n,
            "p": self.env.action_space_pl.nvec[0].item(),
        }

        # For the input space
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

        # Convert the string to dict
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection

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
        self.grammar_planner = {
            key: grammar_planner for key in range(7)
        }

    def build_ppo(
        self,
        rollout_fragment_length,
        batch_size,
        episode_len,
        episodes,
        k_epochs,
        eps_clip,
        gamma,
        device,
        learning_rate,
        _c1,
        _c2,
        mapped_agents,
        obs,
    ):
        """
        Setup specific parameters for PPO
        """
        self.rollout_fragment_length = rollout_fragment_length
        self.batch_size = (
            batch_size if batch_size is not None else episode_len * episodes
        )
        self.step = episode_len
        self.batch_iterations = self.batch_size // self.rollout_fragment_length
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.device = device
        self.learning_rate = learning_rate
        self._c1 = _c1
        self._c2 = _c2

        self.agent = PpoPolicy(self.env.observation_space, 50, K_epochs=self.batch_iterations)
        self.agent.load_model("experiments/" + mapped_agents["a"] + "/models/a_best.pt")

        self.memory = RolloutBuffer(obs, None)
        self.rolling_memory = RolloutBuffer(obs, None)

    def set_seed(self, seed):
        """
        Seeding. Manually set seed for the whole project.
        """
        assert seed >= 1, "Seed must be greater than 0"
        torch.manual_seed(seed)
        np.random.seed(seed=seed)
        random.seed(seed)
        self.env.seed(seed)
        self.seed = seed

    def validate_config_ppo(self):
        """
        Validate PPO's config.

        Raise:
            ValueError if a specific parameter is not set correctly.
        """
        if self.step < 0:
            raise ValueError("'step' must be > 0!")

        if not isinstance(self.seed, int):
            raise ValueError("'seed' muse be integer!")

        if self.k_epochs < 0:
            raise ValueError("'k_epochs' must be > 0!")

        if self.eps_clip < 0:
            raise ValueError("'eps_clip' must be > 0!")

        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("'gamma' must be between (0,1).")

        if self.device != "cpu":
            # raise ValueError()
            logging.warning(
                "The only available 'device' at the moment is 'cpu'. Redirecting everything to 'cpu'!"
            )
            self.device = "cpu"

        if self.learning_rate < 0 and self.learning_rate > 1:
            raise ValueError("'learning_rate' must be between (0,1).")

        logging.debug("Configuration validated: OK")
        return

    def validate_config_dt(self):
        """
        Checks if all variables are set.
        """
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

    def create_directories_and_log_files(self, agent, planner):
        """
        Create directories for the experiment and initialize log files
        (like .csv headers)
        """
        phase = "P1" if agent and not planner else "P2"
        date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logdir = f"experiments/PPO_DT_{phase}_{date}_{self.episodes}"
        self.logfile = os.path.join(self.logdir, "log.txt")

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

        with open(f"{self.logdir}/rewards/ppo.csv", "w") as f:
            f.write("0,1,2,3,p\n")

        with open(
                os.path.join(self.logdir, "losses.csv"), "w"
            ) as f:
            f.write("total,actor,critic,entropy\n")

    def evaluate_fitness(
        self,
        genotypes: List[int],
    ) -> float:
        phenotypes = []
        for idx, genotype in enumerate(genotypes):
            if isinstance(genotype, int):
                genotype = np.array([genotype])
            phenotype, _ = GrammaticalEvolutionTranslator(self.grammar_planner[idx]).genotype_to_str(genotype)
            phenotypes.append(phenotype)

        dt_p = ForestTree(
            phenotypes,
            self.n_actions.get("p", 22),
            self.lr,
            self.df,
            self.eps,
            self.low,
            self.up,
        )

        # Evaluate the fitness
        return self.fitness(dt_p)

    def batch(self, planner):
        """
        Creates a batch of `rollout_fragment_length` steps,
        """
        obs = self.env.reset()

        self.memory.clear()

        steps = 0
        for i in range(self.batch_iterations):
            for j in range(self.rollout_fragment_length):
                # get actions, action_logprob for all agents in each policy* wrt observation
                policy_action, policy_logprob = self.get_actions(obs, planner)

                next_obs, rew, done, _ = self.env.step(policy_action)

                if done["__all__"] is True:
                    next_obs = self.env.reset()

                policy_action["p"] = np.zeros((1))

                self.rolling_memory.update(
                    action=policy_action,
                    logprob=policy_logprob,
                    state=obs,
                    reward=rew,
                    is_terminal=done["__all__"],
                )

                obs = next_obs
            steps += self.rollout_fragment_length

            if steps >= 1000:
                # reset batching environment and get its observation
                obs = self.env.reset()
                steps = 0
            
            self.memory.extend(self.rolling_memory)
            self.rolling_memory.clear()

        data = []
        data.append((self.memory.buffers["0"].rewards.sum()).item())
        data.append((self.memory.buffers["1"].rewards.sum()).item())
        data.append((self.memory.buffers["2"].rewards.sum()).item())
        data.append((self.memory.buffers["3"].rewards.sum()).item())
        data.append((self.memory.buffers["p"].rewards.sum()).item())
        
        data1 = f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]}\n"
        with open(f"{self.logdir}/rewards/ppo.csv", "a+") as file:
            file.write(data1)

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

    def get_actions(self, obs, planner):
        policy_probs = None
        policy_actions = []
        for x in ["0", "1", "2", "3"]:
            a, b = self.agent.act(obs[x])
            policy_actions.append(a)
            policy_probs = self.__append_tensor(policy_probs, b)

        actions = {
            "0": policy_actions[0],
            "1": policy_actions[1],
            "2": policy_actions[2],
            "3": policy_actions[3],
            "p": planner,
        }

        probs = {
            "0": policy_probs[0],
            "1": policy_probs[1],
            "2": policy_probs[2],
            "3": policy_probs[3],
            "p": np.zeros((1)),
        }

        return actions, probs

    def get_actions_ppo_only(self, obs):
        policy_actions = []
        for x in ["0", "1", "2", "3"]:
            a, _ = self.agent.act(obs[x])
            policy_actions.append(a)

        actions = {
            "0": policy_actions[0],
            "1": policy_actions[1],
            "2": policy_actions[2],
            "3": policy_actions[3],
        }

        return actions

    def stepper(
        self, agent_path: str, planner_path: str = None, env: EnvWrapper = None
    ):
        """
        Stepper used for the `interact.py`. N.B.: In both agent and planner paths the prefix `experiments/` is already added.

        ---

        Parameters:

        agent_path: str
            Path to the agent's folder.
        planner_path: str
            Path to the planner's folder.
        env: BaseEnvironment
            Environment used for the interaction.

        ---

        Returns:

        Tuple[List[float], Dict[str, np.ndarray]]
            Tuple containing the list of the rewards and the dictionary containing the
            observations.

        ---

        """

        _agent_path = os.path.join(agent_path, "models", "a_best.pt")
        if not os.path.exists(_agent_path):
            _agent_path = os.path.join(
                "experiments", "PPO_P1", "models", "a_best.pt"
            )
            if not os.path.exists(_agent_path):
                raise FileNotFoundError(
                    f"No agent found neither in '{agent_path}' nor in 'PPO_P1'."
                )

        agent: PytorchLinearA = torch.load(_agent_path)
        planner = ForestTree(#PythonDT(
            load_path=os.path.join(planner_path, "models")
        )

        # Initialize some variables
        obs: Dict[str, Dict[str, np.ndarray]] = env.reset(force_dense_logging=True)
        # planner.new_episode()

        # Run the episode
        for t in range(env.env.episode_length):
            with torch.no_grad():
                actions = agent.get_actions(obs)
            actions["p"] = planner(obs.get("p").get("flat"))

            obs, rew, done, _ = env.step(actions)
            planner.add_rewards(rewards=rew)
            planner.set_reward(rew.get("p", 0))

            if done["__all__"] is True:
                break

        if done["__all__"].item() is True:
            print("Episode finished")
            return planner.rewards, env.env.previous_episode_dense_log
            # Start the episode
            # planner.new_episode()

            # planner_action = planner(obs.get("p").get("flat"))
            # actions = {key: [0] for key in obs.keys()}
            # actions["p"] = planner_action
            # obs, rew, done, _ = env.step(actions)

            # planner.set_reward(rew.get("p", 0))

        return planner.rewards, env.env.previous_episode_dense_log

    def fitness(self, planner: ForestTree):#PythonDT):
        temp = []
        global_cumulative_rewards = []

        if any([not v for v in planner.leaves.values()]):
            return (-np.inf,), planner

        for _ in range(self.episodes):
            # learn on DT
            # Set the seed and reset the environment
            self.seed += 1
            self.env.seed = self.seed
            obs: Dict[str, Dict[str, np.ndarray]] = self.env.reset()

            # Initialize some variables
            cum_global_rew = 0
            cum_rew = {key: np.empty((0)) for key in obs.keys()}
            # static - one per epoch - action by the planner

            planner.new_episode()
            # Run the episode

            for t in range(self.episode_len):
                if t > 1000:
                    obs = self.env.reset()
                # Start the episode
                actions = self.get_actions_ppo_only(obs)
                actions['p'] = planner(obs.get("p").get("flat"))

                if any([action is None for action in actions.values()]):
                    raise ValueError(f"Actions is None: {actions}")

                obs, rew, done, _ = self.env.step(actions)

                for key in obs.keys():
                    cum_rew[key] = np.concatenate((cum_rew[key], rew.get(key, np.nan)))

                cum_global_rew += np.sum([rew.get(k, 0) for k in rew.keys()])
                temp.append(rew.get('p', np.nan))

                if done["__all__"].item() is True:
                    obs = self.env.reset()

                planner.set_reward(rew.get("p", 0))

            # Original version - global cumulative rewards
            # global_cumulative_rewards.append(cum_global_rew)
            # New version - cumulative rewards planner only
            global_cumulative_rewards.append(np.sum(temp))
            temp = []

            # Add the rewards to the planner
            new_rewards = {}
            for key in obs.keys():
                new_rewards[key] = np.sum(cum_rew.get(key, np.nan))

            planner.add_rewards(rewards=new_rewards)

        fitness = (np.max(global_cumulative_rewards),)
        
        return fitness, planner

    def train(self) -> None:
        pop, log, hof, best_leaves = self.grammatical_evolution(
            fitness_function=self.evaluate_fitness,
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

        self.training_log(pop, log, hof, best_leaves)

        return

    def training_log(self, pop, log, hof, best_leaves) -> None:
        """
        Training log. Use this only from `self.train()`
        """
        with open(self.logfile, "a") as log_:
            log_.write("\n\n" + str(log) + "\n")
            log_.write("Best fitness: {}\n".format(hof[0].fitness.values[0]))
            for a, leaves in best_leaves.items():
                phenotype, _ = GrammaticalEvolutionTranslator(self.grammar_planner[a]).genotype_to_str(hof[0][a])
                phenotype = phenotype.replace('leaf="_leaf"', '')
                for k in range(50000):  # Iterate over all possible leaves
                    key = "leaf_{}".format(k)
                    if key in leaves:
                        v = leaves[key].q
                        phenotype = phenotype.replace("out=_leaf", "out={}".format(np.argmax(v)), 1)
                    else:
                        break

                log_.write("\n\t - DT {} - \n\n".format(a+1) + str(hof[0][a]) + "\n\n")
                log_.write(phenotype + "\n\n")
                

        with open(os.path.join(self.logdir, "fitness.tsv"), "w") as f:
            f.write(str(log))
            
        return

    # def train(
    #     self,
    # ) -> None:
    #     with parallel_backend("multiprocessing"):
    #         pop, log, hof, best_leaves = self.grammatical_evolution(
    #             self.evaluate_fitness,
    #             individuals=self.lambda_,
    #             generations=self.generations,
    #             cx_prob=self.cxp,
    #             m_prob=self.mp,
    #             logfile=self.logfile,
    #             mutation=self.mutation,
    #             crossover=self.crossover,
    #             initial_len=self.genotype_len,
    #             selection=self.selection,
    #             planner_only=True,
    #         )

    #     with open(self.logfile, "a") as log_:
    #         phenotype, _ = GrammaticalEvolutionTranslator(self.grammar_planner).genotype_to_str(hof[0])
    #         phenotype = phenotype.replace('leaf="_leaf"', '')

    #         for k in range(50000):  # Iterate over all possible leaves
    #             key = "leaf_{}".format(k)
    #             if key in best_leaves:
    #                 v = best_leaves[key].q
    #                 phenotype = phenotype.replace("out=_leaf", "out={}".format(np.argmax(v)), 1)
    #             else:
    #                 break

    #         log_.write(str(log) + "\n")
    #         log_.write(str(hof[0]) + "\n")
    #         log_.write(phenotype + "\n")
    #         log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))


    #     with open(os.path.join(self.logdir, "fitness.tsv"), "+w") as f:
    #         f.write(str(log))


    def grammatical_evolution(
        self,
        fitness_function,
        individuals,
        generations,
        cx_prob,
        m_prob,
        initial_len=100,
        selection={"function": "tools.selBest"},
        mutation={"function": "ge_mutate", "attribute": None},
        crossover={"function": "ge_mate", "individual": None},
        logfile=None,
        planner_only:bool = False,
    ):
        # random.seed(seed)
        # np.random.seed(seed)

        _max_value = 40000

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(
            "Individual", ListWithParents, typecode="d", fitness=creator.FitnessMax
        )

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_bool", random.randint, 0, _max_value)
        toolbox.register(
            "sub_individuals", 
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            initial_len
        )

        # Structure initializers
        # if jobs > 1:
        #     toolbox.register("map", get_map(jobs, timeout))
            # toolbox.register("map", multiprocess.Pool(jobs).map)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.sub_individuals,
            7,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", fitness_function)

        for d in [mutation, crossover]:
            if "attribute" in d:
                d["attribute"] = toolbox.attr_bool
            if "individual" in d:
                d["individual"] = creator.Individual

        toolbox.register(
            "mate",
            eval(crossover["function"]),
            **{k: v for k, v in crossover.items() if k != "function"},
        )
        toolbox.register(
            "mutate",
            eval(mutation["function"]),
            **{k: v for k, v in mutation.items() if k != "function"},
        )
        toolbox.register(
            "select",
            eval(selection["function"]),
            **{k: v for k, v in selection.items() if k != "function"},
        )

        pop = toolbox.population(n=individuals)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log, best_leaves = self.eaSimplePlanner(
            pop,
            toolbox,
            cxpb=cx_prob,
            mutpb=m_prob,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
            logfile=logfile,
        )

        return pop, log, hof, best_leaves

    def eaSimplePlanner(
        self,
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        verbose=__debug__,
        logfile=None,
        var=varAnd,
    ):
        """This algorithm reproduce the simplest evolutionary algorithm as
        presented in chapter 7 of [Back2000]_.

        :param population: A list of individuals.
        :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                        operators.
        :param cxpb: The probability of mating two individuals.
        :param mutpb: The probability of mutating an individual.
        :param ngen: The number of generation.
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                    inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                        contain the best individuals, optional.
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :returns: A class:`~deap.tools.Logbook` with the statistics of the
                evolution

        The algorithm takes in a population and evolves it in place using the
        :meth:`varAnd` method. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evaluations for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The *cxpb* and *mutpb* arguments are passed to the
        :func:`varAnd` function. The pseudocode goes as follow ::

            evaluate(population)
            for g in range(ngen):
                population = select(population, len(population))
                offspring = varAnd(population, toolbox, cxpb, mutpb)
                evaluate(offspring)
                population = offspring

        As stated in the pseudocode above, the algorithm goes as follow. First, it
        evaluates the individuals with an invalid fitness. Second, it enters the
        generational loop where the selection procedure is applied to entirely
        replace the parental population. The 1:1 replacement ratio of this
        algorithm **requires** the selection procedure to be stochastic and to
        select multiple times the same individual, for example,
        :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
        Third, it applies the :func:`varAnd` function to produce the next
        generation population. Fourth, it evaluates the new individuals and
        compute the statistics on this population. Finally, when *ngen*
        generations are done, the algorithm returns a tuple with the final
        population and a :class:`~deap.tools.Logbook` of the evolution.

        .. note::

            Using a non-stochastic selection method will result in no selection as
            the operator selects *n* individuals from a pool of *n*.

        This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
        :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
        registered in the toolbox.

        .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
        Basic Algorithms and Operators", 2000.
        """
        logbook = tools.Logbook()
        logbook.header = ["time","gen", "nevals", "loss"] + (stats.fields if stats else [])
        best = None
        best_leaves = None
        best_planner: ForestTree = None
        best_actor_loss = np.inf

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid] 

        fitnesses: List[float, ForestTree] = [
            *toolbox.map(toolbox.evaluate, invalid_ind)
        ]  # Obviously, it depends directly on the population size!

        new_fitnesses = []
        planners = []
        for fit, planner in fitnesses:
            new_fitnesses.append(fit)
            planners.append(planner)
        fitnesses = new_fitnesses
        for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            ind.fitness.values = fit
            if logfile is not None and (best is None or best < fit[0]):
                best = fit[0]
                best_planner = planners[i]
                best_leaves = best_planner.leaves
                best_rewards = best_planner.get_rewards()  
                with open(logfile, "a") as log_:
                    log_.write(
                        "[{:.3f}] New best at generation 0 with fitness {}\n".format(
                            datetime.now(), best
                        )
                    )
                    log_.write(str(ind) + "\n")
                    log_.write("Planner Leaves\n")
                    log_.write(str(best_planner.leaves) + "\n")

                best_planner.save(
                    save_path=os.path.join(self.logdir, "models")
                )

        # Save rewards
        with open(os.path.join(self.logdir, "rewards", "dt.csv"), "a") as f:
            f.write(f"{best_rewards['0']},{best_rewards['1']},{best_rewards['2']},{best_rewards['3']},{best_rewards['p']},{np.array([f[0] for f in fitnesses], dtype=np.float16)}\n")

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(
            time=datetime.now().strftime("%H:%M:%S"),
            gen=0,
            nevals=len(invalid_ind),
            loss=(-np.inf, -np.inf, -np.inf, -np.inf),
            **record,
        )
        if verbose:
            print(logbook.stream)

        try:
            # Begin the generational process
            for gen in range(1, ngen):
                # pa is the planner action
                obs = self.env.reset()
                pa = best_planner(obs.get("p").get("flat"))
                self.batch(pa)

                # Select the next generation individuals
                offspring = toolbox.select(population, len(population))

                # Vary the pool of individuals
                offspring = var(offspring, toolbox, cxpb, mutpb)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses: List[float, ForestTree] = [
                    *toolbox.map(toolbox.evaluate, invalid_ind)
                ]

                new_fitnesses = []
                planners = []
                for fit, planner in fitnesses:
                    planners.append(planner)
                    new_fitnesses.append(fit)
                fitnesses = new_fitnesses

                # learn on PPO
                data = self.agent.learn(self.memory.to_tensor()["a"])

                actor_loss = data.get('a_loss', 0).item()
                critic_loss = data.get('c_loss', 0).item()
                entropy = data.get('entropy', 0).item()
                total_loss = data.get('loss', 0)

                with open(
                    os.path.join(self.logdir, "losses.csv"), "a+"
                ) as f:
                    f.write(f"{total_loss},{actor_loss},{critic_loss},{entropy}\n")

                if total_loss < best_actor_loss:
                    self.agent.save_model(os.path.join(self.logdir, "models", "a_best.pt"))
                    best_actor_loss = total_loss

                for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
                    ind.fitness.values = fit
                    if logfile is not None and (best is None or best < fit[0]):
                        best = fit[0]
                        best_planner = planners[i]
                        best_leaves = best_planner.leaves
                        best_rewards = best_planner.get_rewards() 
                        with open(logfile, "a") as log_:
                            log_.write(
                                "[{}] New best at generation {} with fitness {}\n".format(
                                    datetime.now(), gen, fit
                                )
                            )
                            log_.write(str(ind) + "\n")
                            log_.write("Planner leaves\n")
                            log_.write(str(best_planner) + "\n")

                        best_planner.save(
                            save_path=os.path.join(self.logdir, "models")
                        )

                        self.agent.save_model(os.path.join(self.logdir, "models", "a_dt_best.pt"))
                        best_actor_loss = actor_loss

                # Save rewards
                with open(
                    os.path.join(self.logdir, "rewards", "dt.csv"), "a"
                ) as f:
                    f.write(f"{best_rewards['0']},{best_rewards['1']},{best_rewards['2']},{best_rewards['3']},{best_rewards['p']},{np.array([f[0] for f in fitnesses], dtype=np.float16)}\n")

                # Update the hall of fame with the generated individuals
                if halloffame is not None:
                    halloffame.update(offspring)

                # Replace the current population by the offspring
                for o in offspring:
                    argmin = np.argmin(
                        map(lambda x: population[x].fitness.values[0], o.parents)
                    )

                    if o.fitness.values[0] > population[o.parents[argmin]].fitness.values[0]:
                        population[o.parents[argmin]] = o

                # Append the current generation statistics to the logbook
                record = stats.compile(population) if stats else {}
                loss_str = f"({round(total_loss,2) if not np.isinf(total_loss) else total_loss},{round(actor_loss,2) if not np.isinf(actor_loss) else actor_loss},{round(critic_loss,2) if not np.isinf(critic_loss) else critic_loss},{round(entropy,2) if not np.isinf(entropy) else entropy})"
                logbook.record(
                    time=datetime.now().strftime("%H:%M:%S"),
                    gen=gen,
                    nevals=len(invalid_ind),
                    loss=loss_str,
                    **record,
                )
                if verbose:
                    print(logbook.stream)

        except KeyboardInterrupt:
            logging.warning(f"KeyboardInterrupt occurred! Exiting...")

        return population, logbook, best_leaves
