# Rl Agent for Crypto Currncy Trade

Agent trained using DQN und DRL to Trade Crypto Currency.

## Prep data folder for Evaluation notebook
[Link to training data](https://tubcloud.tu-berlin.de/s/cKyHWpT6w9J32oa)
 - Extract all trainings to one folder. It can be any folder, just set the variable "base_folder" as the full path to that folder.

 - In the Evaluation Notebook (Compare results DRL DQN.ipynb) set folder path to var base_folder

Folder should look like:

base_folder/
| 20230429_200721
| 20230424_070731
| ...


## Best Trainings

DRL_TRAININGS = {
    "reward_sharpe_ratio_0": "20230429_200721",
    "reward_differential_sharpe_ratio_0": "20230424_070731",
    "compute_reward_from_tutor_0": "20230429_200559",
    "reward_profit_0": "20230429_110440",
    "reward_profit_1": "20230423_174023",
    "reward_profit_2": "20230420_083508", # trial 8
    "reward_profit_3": "20230420_195053", # trial 9
}

DQN_TRAININGS = {
    "reward_sharpe_ratio_0": "20230427_165557",
    "reward_differential_sharpe_ratio_0": "20230427_083418",
    "compute_reward_from_tutor_0": "20230427_083632",
    "reward_profit_0": "20230425_145617",

    "reward_differential_sharpe_ratio_1": "20230423_114422",
    "compute_reward_from_tutor_1": "20230421_181519",
    "reward_profit_1": "20230423_231505",
    "reward_profit_2": "20230422_122001",
    "reward_profit_3": "20230421_001115",
}


