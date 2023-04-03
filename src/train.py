import sys
import os
from datetime import datetime
import json
import shutil
import random
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# tf.compat.v1.disable_eager_execution()
from tqdm import tqdm_notebook, tqdm
from matplotlib import pyplot as plt
import pprint
import gc

sys.path.append('../src/')
from env import BTCMarket_Env
from agent import Trader_Agent
from DQNtrainer import DQNTrainer
from DRLtrainer import DRLTrainer
from collections import deque
import h5py
from itertools import product

def enable_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUS: {gpus}")
    if gpus:
        # Restrict TensorFlow to only allocate 3GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def conduct_traning(param_combination, i):
    if param_combination.get('trainer') == 'DQNTrainer':
        action_space = 4
        algorithm = f'DQN_trial_{i}'
    else:
        action_space = 1
        algorithm = f'DRL_trial_{i}'

    if param_combination.get('asset') == 'BTC':
        param_combination["data_source"] = data_source_btc
    else:
        param_combination["data_source"] = data_source_perp


    param_combination['action_space'] = action_space
    param_combination['algorithm'] = algorithm
    pprint.pprint(param_combination)

    env = BTCMarket_Env(observation_space = param_combination.get('obs_space', (8,20)),
            action_space = action_space,
            start_money = param_combination.get('money', 10000),
            trading_fee = param_combination.get('fee', 0.001),
            asset = param_combination.get('asset', 'BTC'),
            source_file = param_combination.get('data_source', data_source_btc),
            reward_function = param_combination.get('reward_function', 'reward_differential_sharpe_ratio'),
                       )

    agent = Trader_Agent(observation_space = param_combination.get('obs_space', (8,20)),
                action_space = action_space,
                action_domain = param_combination.get('action_domain', (0.0,1.0)),
                epsilon = param_combination.get('epsilon', 0.7),
                epsilon_final = param_combination.get('epsilon_final', 0.01),
                epsilon_decay = param_combination.get('epsilon_decay', 0.995),
                        )

    trainer_class = eval(param_combination.get('trainer', 'DRLTrainer'))
    trainer = trainer_class(env, agent,
                observation_space = param_combination.get('obs_space', (8,20)),
                action_space = action_space,
                batch_size=param_combination.get('batch_size', 50),
                epoch=param_combination.get('epoch', 5),
                gamma=param_combination.get('gamma', 0.95),
                learning_rate=param_combination.get('learning_rate', 1e-3),
                algorithm=algorithm,
                lstm_path="./../notebooks/best_models/11_mar_2023/best_model_sequential_20back_10ahead.h5",
                # best_model_sequential_20back_10ahead lstm_2,
                           )

    os.makedirs(trainer.train_folder,exist_ok=True)
    with open(f'{trainer.train_folder}/params.json', 'w') as fp:
        json.dump(param_combination, fp)

    trainer.rollout(n_episodes=param_combination.get('episodes', 2), 
                       run_per_episode=param_combination.get('runs_p_eps', 2))
    
    return trainer.train_folder 

if __name__=='__main__':
    enable_memory_growth()

    training_folders = []

    data_source_btc = "BTC_histData_dt1800.0s_20220825_0629" 
    data_source_perp = "Perp_BTC_FundingRate_Data_fakehist"

    hpo_params= { 
        'obs_space' : [(8,20)], 
        'action_domain' : [(0.0,1.0)], # (-1.0,1.0),
        'money' : [10000], 
        'fee' : [0.001], 
        'asset' : ['BTC'],
        'reward_function' : ['compute_reward_from_tutor', 'reward_sharpe_ratio', 'reward_sortino_ratio', 
                            'reward_differential_sharpe_ratio',],
        #     'reward_function' : ['reward_sterling_ratio'],

        'learning_rate': [1e-3],
        'trainer' : ['DQNTrainer', 'DRLTrainer'], # 'DRLTrainer', 'DQNTrainer'
        'episodes' : [50], 
        'runs_p_eps' : [5], 
        'batch_size': [1],
        'epoch': [5],
        'gamma': [0.95],

        'epsilon': [0.7],
        'epsilon_final':[0.01],
        'epsilon_decay':[0.75],
        }


    keys, values = zip(*hpo_params.items())
    hpo_list = [dict(zip(keys, v)) for v in product(*values)]
    # pprint.pprint(hpo_list)
    
    # for i , params in enumerate(hpo_params):

    i = 0
    params = {'action_domain': (0.0, 1.0),
                'action_space': 4,
                'algorithm': 'DRL_trial_0',
                'asset': 'BTC',
                'batch_size': 1,
                'data_source': 'BTC_histData_dt1800.0s_20220825_0629',
                'episodes': 50,
                'epoch': 5,
                'epsilon': 0.5,
                'epsilon_decay': 0.75,
                'epsilon_final': 0.01,
                'fee': 0.001,
                'gamma': 0.95,
                'learning_rate': 0.001,
                'money': 10000,
                'obs_space': (8, 20),
                'reward_function': 'compute_reward_from_tutor',
                'runs_p_eps': 5,
                'DQNTrainer': 'DRLTrainer' # DRLTrainer DQNTrainer
                }

    train_folder = conduct_traning(params, i)

    # Plot profit  
    # train_folder = './../data/20230328_175512/DQN_trial_0/'
    print(len(hpo_list))
    training_folders.append(train_folder)
    df = pd.read_csv(f'{train_folder}/Trainer_Data.csv')
    print(df.shape)
    print(max(df.time_elapsed))

    fig, ax = plt.subplots(2,1, figsize=(6,6))
    ax[0].plot(df.profit)
    ax[0].set_title("Profits")
    y = [ast.literal_eval(x)[0] for x  in df.reward]
    ax[1].plot(y)
    ax[1].set_title("Rewards")
    plt.show()
    plt.savefig(f'{train_folder}/profit_rewards.png')
