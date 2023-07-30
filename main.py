"""
Main file to run the experiments.

args:
    --mode: train or eval
    --type: PPO, DT or PPO_DT
    --path-ppo: path of the model weights for ppo. If None or True, the ppo agents are trained from scratch. If False, they are not considered at all.
    --path-dt: path of the model weights for dt. If None or True, the decision tree is trained from scratch. If False, it is not considered at all.

Author(s): bonom, Sa1g
Github(s): https://github.com/bonom, https://github.com/sa1g

Date: 2023.07.30
    
"""
import os
import logging
import argparse

from src.common import get_environment
from src.train.ppo_dt import PPODtTrainConfig
from src import DtTrainConfig, InteractConfig, PpoTrainConfig

def get_mapping_function():
    """
    Returns a mapping function given the agent key.
    """
    def mapping_function(key: str) -> str:
        """
        - `a` -> economic player
        - `p` -> social planner
        """
        if str(key).isdigit() or key == "a":
            return "a"
        return "p"

    return mapping_function

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="Mode of the experiment")
parser.add_argument("--type", type=str, default="PPO", help="Type of the algorithm")
parser.add_argument(
    "--path-ppo", type=str, default=True, help="Path of the model weights for ppo"
)
parser.add_argument(
    "--path-dt", type=str, default=False, help="Path of the model weights for dt"
)

args = parser.parse_args()

# Check arguments
assert args.mode in ["train", "eval"], "Invalid mode"
assert args.type in ["PPO", "DT", "PPO_DT"], "Invalid type of algorithm"
assert args.path_ppo is not None or args.path_dt is not None, "Path of the model weights is not set"

assert isinstance(args.path_ppo, str) or isinstance(args.path_ppo, bool), "PPO path must be either a path like object or a bool object!"
if isinstance(args.path_ppo, str):
    if args.path_ppo in ['True', 'False']:
        if args.path_ppo == 'True':
            args.path_ppo = True
        else:
            args.path_ppo = False
    else:
        assert os.path.exists(os.path.join('experiments', args.path_ppo)), "PPO path not found ('{}')".format(args.path_ppo)

assert isinstance(args.path_dt, str) or isinstance(args.path_dt, bool), "DT path must be either a path like object or a bool object!"
if isinstance(args.path_dt, str):
    if args.path_dt in ['True', 'False']:
        if args.path_dt == 'True':
            args.path_dt = True
        else:
            args.path_dt = False
    else:
        assert os.path.exists(os.path.join('experiments', args.path_dt)), "DT path not found ('{}')".format(args.path_dt)

# Run experiment
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(process)d - %(asctime)s - %(levelname)s - %(message)s",
    )

    # Get environment
    env = get_environment()

    # Run training or evaluation
    if args.mode == "train":
        if args.type == "PPO":
            trainer = PpoTrainConfig(
                get_mapping_function,
                env,
                num_workers=18,
                step=10000,
                batch_size=4000,
                rollout_fragment_length=200,
                k_epochs=40,
                seed=1,
                mapped_agents={
                    "a": args.path_ppo,
                    "p": args.path_dt
                },
            )
            trainer.train()
        elif args.type == "DT":
            trainer = DtTrainConfig(
                env,
                episodes=5,
                episode_len=1000,
                lambda_=180,
                generations=50,
                mapped_agents={
                    "a": args.path_ppo,
                    "p": args.path_dt
                },
            )
            trainer.train()
        elif args.type == "PPO_DT":
            trainer = PPODtTrainConfig(
                env,
                episodes=1,
                episode_len=5000,
                lambda_=40,
                generations=2000,
                seed=1,
                mapped_agents={
                    "a": args.path_ppo,  # This MUST be the folder name to load the agent pre-trained in pytorch
                    "p": args.path_dt
                }
            )
            trainer.train()
        else:
            raise ValueError("Invalid type of algorithm")

    elif args.mode == "eval":
        # When finished, evaluate the agent
        if args.type == "PPO":
            interact = InteractConfig(
                get_mapping_function,
                env,
                PpoTrainConfig,
                config={},
                mapped_agents={
                    "a": args.path_ppo, 
                    "p": args.path_dt,
                },
            )
        elif args.type == "DT":
            interact = InteractConfig(
                get_mapping_function,
                env,
                DtTrainConfig,
                config={},
                mapped_agents={
                    "a": args.path_ppo, 
                    "p": args.path_dt,
                },
            )
        elif args.type == "PPO_DT":
            interact = InteractConfig(
                get_mapping_function,
                env,
                PPODtTrainConfig,
                config={},
                mapped_agents={
                    "a": args.path_ppo, 
                    "p": args.path_dt,
                },
            )

        # Run the interaction
        dense_logs = interact.run_stepper()
        # Plot the results
        interact.output_plots(dense_logs)
    else:
        raise ValueError("Invalid mode")
    