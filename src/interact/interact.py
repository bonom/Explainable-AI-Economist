"""
Interact module. 

=================

This module is intended to be used to interact with the environment using the pretrained models. It selects the right configuration
and performs a number of steps in the environment. It also saves the dense log of the episode and the rewards in a csv file.

---

Author(s): bonom, Sa1g
Github(s): https://github.com/bonom, https://github.com/sa1g

Date: 2023.07.30

"""
import datetime
import os
import torch
import pickle
import random
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.interact import plotting
from src.train.ppo_dt import PPODtTrainConfig
from src.train.ppo_train import PpoTrainConfig
from src.train.dt_ql_train import DtTrainConfig
from src.common import EmptyModel, test_mapping


class InteractConfig:
    """
    Endpoint to setup interact configuration.

    Args:
        env
        trainer: it can be one between DtTrainConfig and PpoTrainConfig
        steps: how many steps to step the environment with these pretrained models
        config: TODO
        mapped_agents: the same as in PpoTrainConfig
    """

    def __init__(
        self,
        mapping_function,
        env,
        trainer,
        config: dict,
        mapped_agents: dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 1,
    ):
        self.mapping_function = mapping_function
        self.env = env
        self.device = device
        self.trainer = trainer
        self.seed = seed
        self.config = config
        self.mapped_agents = mapped_agents

        self.validate_config(env=env)
        self.setup_logs_and_dirs()

        return        

    def run_stepper(self):
        if self.trainer == PpoTrainConfig:
            # PPO
            if self.phase == "P1":
                self.models = {
                    "a": torch.load(
                        "experiments/" + self.mapped_agents.get("a") + "/models/a_best.pt"
                    ),
                    "p": EmptyModel(
                        self.env.observation_space_pl,
                        7,
                    ),
                }
            else:
                self.models = {
                    "a": torch.load(
                        "experiments/" + self.mapped_agents["a"] + "/models/a.pt"
                    ),
                    "p": torch.load(
                        "experiments/" + self.mapped_agents["p"] + "/models/p.pt"
                    ),
                }

            def stepper():
                def get_actions(obs: dict):
                    actions = {}
                    for key in ['0', '1', '2', '3']:
                        actions[key], _ = self.models[self.mapping_function(key)].act(
                            obs[key]
                        )
                    actions['p'] = np.zeros((7))

                    return actions

                env = self.env
                env.seed = self.seed
                obs = env.reset(force_dense_logging=True)

                for _ in tqdm(range(env.env.episode_length)):
                    actions = get_actions(obs)
                    obs, rew, done, _ = env.step(actions)

                    if done["__all__"] is True:
                        break

                    with open(self.path + "/logs/Simulation.csv", "a") as log_file:
                        log_file.write(
                            f"{rew['0']},{rew['1']},{rew['2']},{rew['3']},{rew['p']}\n"
                        )

                dense_log = env.env.previous_episode_dense_log

                with open(self.path + "/logs/dense_logs.pkl", "wb") as dense_logs:
                    pickle.dump(dense_log, dense_logs)

                return dense_log

        elif self.trainer == DtTrainConfig:
            if self.phase == "P1":
                self.models = {
                    "a": os.path.join(
                        "experiments", self.mapped_agents.get("a"), "models", "dt_a.pkl"
                    ),
                    "p": None,
                }
            else:
                self.models = {
                    "a": os.path.join(
                        "experiments",
                        self.mapped_agents.get("a"),
                    ),
                    "p": os.path.join(
                        "experiments",
                        self.mapped_agents.get("p"),
                    ),
                }

            def stepper():
                env = self.env
                env.seed = self.seed

                # Done only for intellisense and to remember types
                self.trainer = DtTrainConfig()

                rewards, dense_log = self.trainer.stepper(
                    agent_path=self.models["a"], planner_path=self.models["p"], env=env
                )

                with open(os.path.join(self.path, "logs", "1.csv"), "a") as reward_file:
                    for rew in rewards:
                        reward_file.write(
                            f"{rew['0']},{rew['1']},{rew['2']},{rew['3']},{rew['p']}\n"
                        )

                with open(
                    os.path.join(self.path, "logs", "dense_logs.pkl"), "wb"
                ) as log_file:
                    pickle.dump(dense_log, log_file)

                return dense_log

        elif self.trainer == PPODtTrainConfig:
            if self.phase == "P1":
                self.models = {
                    "a": os.path.join(
                        "experiments", self.mapped_agents.get("a"), "models", "a_best.pkl"
                    ),
                    "p": None,
                }
            else:
                self.models = {
                    "a": os.path.join(
                        "experiments",
                        self.mapped_agents.get("a"),
                    ),
                    "p": os.path.join(
                        "experiments",
                        self.mapped_agents.get("p"),
                    ),
                }

            def stepper():
                env = self.env
                env.seed = self.seed
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)

                # Done only for intellisense and to remember types
                self.trainer = PPODtTrainConfig()

                rewards, dense_log = self.trainer.stepper(
                    agent_path=self.models["a"], planner_path=self.models["p"], env=env
                )

                with open(
                    os.path.join(self.path, "logs", "Simulation.csv"), "a"
                ) as reward_file:
                    for rew in rewards:
                        for val in rew.values():
                            reward_file.write(f"{val.item()},")
                        # reward_file.write(
                        #     f"{rew['0']},{rew['1']},{rew['2']},{rew['3']},{rew['p']}\n"
                        # )
                        reward_file.write("\n")

                with open(
                    os.path.join(self.path, "logs", "dense_logs.pkl"), "wb"
                ) as log_file:
                    pickle.dump(dense_log, log_file)

                return dense_log

        return stepper()

    def setup_logs_and_dirs(self):
        """
        Creates this experiment directory and a log file.
        """

        algorithm_name = "PPO" 
        if self.trainer ==  DtTrainConfig:
            algorithm_name = "DT"
        if self.trainer ==  PPODtTrainConfig:
            algorithm_name = "PPO_DT"

        experiment_name = 'EVAL_{}_{}'.format(algorithm_name, datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        self.path = os.path.join('experiments', experiment_name)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.path + "/logs")
            os.makedirs(self.path + "/plots")

        # Copy all the self.mapped_agents.get("a") content in the new dir
        if self.phase == "P2":
            shutil.copytree(
                os.path.join("experiments", self.mapped_agents.get("p")),
                self.path,
                dirs_exist_ok=True,
            )

        with open(self.path + "/logs/Simulation.csv", "a+") as log_file:
            log_file.write("0,1,2,3,p\n")

        logging.info("Directories created")
        return experiment_name

    def validate_config(self, env):
        """
        Validate PPO's config.

        Raise:
            ValueError if a specific parameter is not set correctly.
        """

        test_mapping(
            mapping_function=self.mapping_function,
            mapped_agents=self.mapped_agents,
            env=env,
        )
        self.mapping_function = self.mapping_function()

        if not isinstance(self.seed, int):
            raise ValueError("'seed' muse be integer!")

        if not self.trainer in [PpoTrainConfig, DtTrainConfig, PPODtTrainConfig]:
            raise ValueError(
                "`self.trainer` must be `PpoTrainConfig` or `DtTrainConfig`!"
            )

        if isinstance(self.mapped_agents["p"], str) or self.mapped_agents.get('p', False):
            self.phase = "P2"
        else:
            self.phase = "P1"

        logging.info("Config validation: OK!")

    def output_plots(self, dense_logs: dict) -> None:
        """
        Plot the rewards and save the plots in the experiment's directory.
        """

        (fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(
            dense_logs
        )

        axes0 = fig0.axes
        axes1 = fig1.axes
        axes2 = fig2.axes

        for i in range(5):
            axes0[i].set_title("Episode {}".format(i*250), fontsize=18)

            axes0[i].set_xlabel("")
            axes0[i].set_ylabel("")

        for i in range(4):           

            axes1[i].set_ylabel("Qty", fontsize=12)
            axes1[i].set_xlabel("Step", fontsize=12)
            axes1[i].tick_params(axis='both', labelsize=12)

            axes2[i].set_title("Movements Agent {}".format(i), fontsize=18)

            axes2[i].set_ylabel("y", fontsize=12)
            axes2[i].set_xlabel("x", fontsize=12)
            axes2[i].tick_params(axis='both', labelsize=12)

            axes2[i+4].set_title("Incomes Agent {}".format(i), fontsize=18)
        
            axes2[i+4].set_ylabel("Income", fontsize=12)
            axes2[i+4].set_xlabel("Step", fontsize=12)
            axes2[i+4].tick_params(axis='both', labelsize=12)

        fig0.tight_layout()
        fig1.tight_layout()
        fig2.tight_layout()

        fig0.savefig(fname=os.path.join(self.path, "plots", "Global.eps"))
        fig0.savefig(fname=os.path.join(self.path, "plots", "Global.png"))

        fig1.savefig(fname=os.path.join(self.path, "plots", "Trend.eps"))
        fig1.savefig(fname=os.path.join(self.path, "plots", "Trend.png"))

        fig2.savefig(fname=os.path.join(self.path, "plots", "Movements.eps"))
        fig2.savefig(fname=os.path.join(self.path, "plots", "Movements.png"))

        with open(
            os.path.join(self.path, "logs", "incomes.pkl"), "+wb"
        ) as incomes_file:
            pickle.dump(incomes, incomes_file)

        with open(os.path.join(self.path, "logs", "endows.pkl"), "+wb") as endows_file:
            pickle.dump(endows, endows_file)

        with open(
            os.path.join(self.path, "logs", "c_trades.pkl"), "+wb"
        ) as c_trades_file:
            pickle.dump(c_trades, c_trades_file)

        with open(
            os.path.join(self.path, "logs", "all_builds.pkl"), "+wb"
        ) as all_builds_file:
            pickle.dump(all_builds, all_builds_file)

        plt.close(fig0)
        plt.close(fig1)
        plt.close(fig2)

        total_tax_paid = 0
        total_income = 0
        total_per_agent = {
            'income' : {
                key: 0 for key in ['0', '1', '2', '3']
            },
            'tax': {
                key: 0 for key in ['0', '1', '2', '3']
            }
        }
        with open(
            os.path.join(self.path, "logs", "PeriodicTax.md"), "+w"
        ) as f:
            if self.phase == 'P1':
                f.write(f"# Periodic Tax\n\n| Step | Agent | Income |\n| --- | --- | --- |\n")
            elif self.phase == 'P2':
                f.write(f"# Periodic Tax\n\n| Step | Agent | Income | Tax | Percentage (Income/Tax * 100) |\n| --- | --- | --- | --- | --- |\n")
            for idx, tax in enumerate(dense_logs.get('PeriodicTax')):
                if len(tax) > 0:
                    for key, values in tax.items():
                        if key in ['0', '1', '2', '3']:
                            _income = values.get('income')
                            _tax_paid = values.get('tax_paid')
                            if self.phase == 'P1':
                                f.write(f"| {idx+1} | {key} | {_income:.2f} \n")
                            elif self.phase == 'P2':
                                if _income <= 0:
                                    f.write(f"| {idx+1} | {key} | {_income:.2f} | {_tax_paid:.2f} | 0.0 % |\n")
                                else:
                                    f.write(f"| {idx+1} | {key} | {_income:.2f} | {_tax_paid:.2f} | {round((_tax_paid/_income)*100, 1)} % |\n")
                            total_per_agent['income'][key] += _income
                            total_per_agent['tax'][key] += _tax_paid
                           
                            total_tax_paid += _tax_paid
                            total_income += _income
            
            if self.phase == 'P1':
                f.write(f"\n\n## Total per agent:\n\n| Agent | Income |\n| --- | --- |\n")
            elif self.phase == 'P2':
                f.write(f"\n\n## Total per agent:\n\n| Agent | Income | Tax | Percentage |\n| --- | --- | --- | --- |\n")
            for agent in ['0', '1', '2', '3']:
                if self.phase == 'P1':
                    f.write(f"| {agent} | {total_per_agent['income'][agent]:.2f} \n")
                elif self.phase == 'P2':
                    f.write(f"| {agent} | {total_per_agent['income'][agent]:.2f} | {total_per_agent['tax'][agent]:.2f} | {round((total_per_agent['tax'][agent]/total_per_agent['income'][agent])*100, 1)} % |\n")
            if self.phase == 'P1':
                f.write(f"\n\n## Total\n\nTo sum up:\n - Total income is {total_income:.2f}\n")
            elif self.phase == 'P2':
                f.write(f"\n\n## Total\n\nTo sum up:\n - Total paid taxes is {total_tax_paid:.2f}\n - Total incomes is {total_income:.2f}\n - Ratio(tax/income) = {round((total_tax_paid/total_income)*100, 2)} %")

        with open(
            os.path.join(self.path, "logs", "PeriodicTax.md"), "a"
        ) as f:
            if self.phase == 'P1':
                f.write(f"# Periodic Tax LaTeX ready\n\n Step & Agent & Income \n")
            elif self.phase == 'P2':
                f.write(f"# Periodic Tax LaTeX ready\n\n Step & Agent & Income & Tax & $\%$ \n")
            for idx, tax in enumerate(dense_logs.get('PeriodicTax')):
                if len(tax) > 0:
                    f.write(f"{idx+1}")
                    for key, values in tax.items():
                        if key in ['0', '1', '2', '3']:
                            _income = values.get('income')
                            _tax_paid = values.get('tax_paid')
                            if self.phase == 'P1':
                                f.write(f" & {key} & {_income:.2f} \n")
                            elif self.phase == 'P2':
                                if _income <= 0:
                                    f.write(f" & {key} & {_income:.2f} & {_tax_paid:.2f} & 0.0 \n")
                                else:
                                    f.write(f" & {key} & {_income:.2f} & {_tax_paid:.2f} & {round((_tax_paid/_income)*100, 1)} \n")
                            total_per_agent['income'][key] += _income
                            total_per_agent['tax'][key] += _tax_paid
                           
                            total_tax_paid += _tax_paid
                            total_income += _income
            
            if self.phase == 'P1':
                f.write(f"\n\n## Total per agent:\n\n Agent & Income \n")
            elif self.phase == 'P2':
                f.write(f"\n\n## Total per agent:\n\n Agent & Income & Tax & Percentage \n")
            for agent in ['0', '1', '2', '3']:
                if self.phase == 'P1':
                    f.write(f" {agent} & {total_per_agent['income'][agent]:.2f} \n")
                elif self.phase == 'P2':
                    f.write(f" {agent} & {total_per_agent['income'][agent]:.2f} & {total_per_agent['tax'][agent]:.2f} & {round((total_per_agent['tax'][agent]/total_per_agent['income'][agent])*100, 1)} \n")
            if self.phase == 'P1':
                f.write(f"\n\n## Total\n\nTo sum up:\n - Total incomes is {total_income:.2f}\n")
            elif self.phase == 'P2':
                f.write(f"\n\n## Total\n\nTo sum up:\n - Total paid taxes is {total_tax_paid:.2f}\n - Total incomes is {total_income:.2f}\n - Ratio(tax/income) = {round((total_tax_paid/total_income)*100, 2)} %")


        agent_1 = np.empty((0))
        agent_2 = np.empty((0))
        agent_3 = np.empty((0))
        agent_4 = np.empty((0))
        planner = np.empty((0))

        for val in dense_logs.get('rewards'):
            agent_1 = np.append(agent_1, val.get('0', -np.inf))
            agent_2 = np.append(agent_2, val.get('1', -np.inf))
            agent_3 = np.append(agent_3, val.get('2', -np.inf))
            agent_4 = np.append(agent_4, val.get('3', -np.inf))
            planner = np.append(planner, val.get('p', -np.inf))

        fig, axs = plt.subplots(3, 2, figsize=(20, 30))

        axs[0][0].scatter(np.arange(1, agent_1.shape[0]+1), agent_1, label='Instant Rewards')
        agent_sum = []
        for rew in agent_1:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[0][0].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[0][0].set_title('Agent 1', fontsize=30)
        axs[0][0].set_xlabel('Episode', fontsize=20)
        axs[0][0].set_ylabel('Reward', fontsize=20)
        axs[0][0].tick_params(axis='x', labelsize=20)
        axs[0][0].tick_params(axis='y', labelsize=20)
        axs[0][0].legend(fontsize=20)

        axs[0][1].scatter(np.arange(1, agent_2.shape[0]+1), agent_2, label='Instant Rewards')
        agent_sum = []
        for rew in agent_2:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[0][1].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[0][1].set_title('Agent 2', fontsize=30)
        axs[0][1].set_xlabel('Episode', fontsize=20)
        axs[0][1].set_ylabel('Reward', fontsize=20)
        axs[0][1].tick_params(axis='x', labelsize=20)
        axs[0][1].tick_params(axis='y', labelsize=20)
        axs[0][1].legend(fontsize=20)

        axs[1][0].scatter(np.arange(1, agent_3.shape[0]+1), agent_3, label='Instant Rewards')
        agent_sum = []
        for rew in agent_3:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[1][0].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[1][0].set_title('Agent 3', fontsize=30)
        axs[1][0].set_xlabel('Episode', fontsize=20)
        axs[1][0].set_ylabel('Reward', fontsize=20)
        axs[1][0].tick_params(axis='x', labelsize=20)
        axs[1][0].tick_params(axis='y', labelsize=20)
        axs[1][0].legend(fontsize=20)

        axs[1][1].scatter(np.arange(1, agent_4.shape[0]+1), agent_4, label='Instant Rewards')
        agent_sum = []
        for rew in agent_4:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[1][1].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[1][1].set_title('Agent 4', fontsize=30)
        axs[1][1].set_xlabel('Episode', fontsize=20)
        axs[1][1].set_ylabel('Reward', fontsize=20)
        axs[1][1].tick_params(axis='x', labelsize=20)
        axs[1][1].tick_params(axis='y', labelsize=20)
        axs[1][1].legend(fontsize=20)

        if self.phase == 'P2':
            axs[2][0].scatter(np.arange(1, planner.shape[0]+1), planner, label='Instant Rewards')
            agent_sum = []
            for rew in planner:
                to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
                agent_sum.append(to_add)
            axs[2][0].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
            axs[2][0].set_title('Planner', fontsize=30)
            axs[2][0].set_xlabel('Episode', fontsize=20)
            axs[2][0].set_ylabel('Reward', fontsize=20)
            axs[2][0].tick_params(axis='x', labelsize=20)
            axs[2][0].tick_params(axis='y', labelsize=20)
            axs[2][0].legend(fontsize=20)

            total = []
            for a1, a2, a3, a4, p in zip(agent_1, agent_2, agent_3, agent_4, planner):
                total.append(a1+a2+a3+a4+p)

            axs[2][1].scatter(np.arange(1, len(total)+1), total, label='Instant Rewards')
            agent_sum = []
            for rew in total:
                to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
                agent_sum.append(to_add)
            axs[2][1].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
            axs[2][1].set_title('Total', fontsize=30)
            axs[2][1].set_xlabel('Episode', fontsize=20)
            axs[2][1].set_ylabel('Reward', fontsize=20)
            axs[2][1].tick_params(axis='x', labelsize=21)
            axs[2][1].tick_params(axis='y', labelsize=21)
            axs[2][1].legend(fontsize=20)

        else:
            total = []
            for a1, a2, a3, a4, p in zip(agent_1, agent_2, agent_3, agent_4, planner):
                total.append(a1+a2+a3+a4+p)

            axs[2][0].scatter(np.arange(1, len(total)+1), total, label='Instant Rewards')
            agent_sum = []
            for rew in total:
                to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
                agent_sum.append(to_add)
            axs[2][0].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
            axs[2][0].set_title('Total', fontsize=30)
            axs[2][0].set_xlabel('Episode', fontsize=20)
            axs[2][0].set_ylabel('Reward', fontsize=20)
            axs[2][0].tick_params(axis='x', labelsize=20)
            axs[2][0].tick_params(axis='y', labelsize=20)
            axs[2][0].legend(fontsize=20)
            
            fig.delaxes(axs[2,1])

        fig.tight_layout()
        fig.savefig(os.path.join(self.path, 'plots', 'Rewards.eps'))
        fig.savefig(os.path.join(self.path, 'plots', 'Rewards.png'))

        plt.close(fig)

        return