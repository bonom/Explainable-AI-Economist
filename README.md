# Explainable AI Economist

Taxation is always a topic of discussion and debate. When considering tax policies there are two extremes: the free market and pure redistribution. The free market considers productivity alone and does not raise any taxes. On the other hand, pure redistribution divides all incomes equally amongst all workers. This should achieves equality but at the potential risk of a large drop in productivity. The aim is to find the best trade-off on the Pareto boundary that connects these two extremes. Thanks to the AI-Economist it is possible to study this topic. However, it is implemented with Reinforcement Learning: a very powerful tool but it lacks of explainability. The aim is to substitute the reinforcement learning social planner with a decision tree one in order to be able to explain how the taxes are applied.

## Getting Started

### Prerequisites

The project is developed in Python 3.7. I strongly recommend to use a virtual environment and install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Training

In order to train the models you can either:

- Train the PPO agents only: 
```bash
python main.py --mode train --type PPO --path-ppo True
```
- Train the Decision Tree with pre-trained PPO agents: 
```bash
python main.py --mode train --type PPO_DT --path-ppo <path name in 'experiments' to the PPO agents> --path-dt True
```
- Train the Decision Tree only for both agents and planner: 
```bash
python main.py --mode train --type DT --path-dt True
```
  
There are other possibilities but they are not recommended. 

The script will save the models and their logs in the `experiments` folder, from which you can load the models for further trainings or evaluations.

### Evaluation

In order to evaluate the models you can either:

- Evaluate the PPO agents only: 
```bash
python main.py --mode eval --type PPO --path-ppo <path name in 'experiments' to the PPO agents>
```

- Evaluate the Decision Tree with pre-trained PPO agents: 
```bash
python main.py --mode eval --type PPO_DT --path-ppo <path name in 'experiments' to the PPO agents> --path-dt <path name in 'experiments' to the Decision Tree>
```

- Evaluate the Decision Tree only for both agents and planner: 
```bash
python main.py --mode eval --type DT --path-dt <path name in 'experiments' to the Decision Tree>
```

The script will save the results and the plots in the `experiments` folder with the string `EVAL_` before the folder path.

## Contributions

Thanks to [Sa1g](https://github.com/sa1g) for writing the the code for an efficient training of the PPO agents.

## License

Foundation, the AI Economist and all the code available in the repository are released under the [BSD-3 License](LICENSE.txt).