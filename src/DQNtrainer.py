import os
from datetime import datetime
import json
import shutil
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# tf.compat.v1.disable_eager_execution()
import gc
from tqdm import tqdm_notebook, tqdm
from matplotlib import pyplot as plt

from env import BTCMarket_Env
from agent import Trader_Agent
from collections import deque

np.random.seed(42)
random.seed(42)

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs=None):
        # Housekeeping
        gc.collect()
        keras.backend.clear_session()


class DQNTrainer():
    '''
    DQN Trainer class

    Reinforcement Algorithm. Trains the agent (Policy Network).
    '''
    def __init__(self, env, agent,
                observation_space: tuple, 
                action_space: int,
                batch_size: int,
                epoch : int = 5, 
                gamma: float = 0.95,
                learning_rate: float =1e-3,
                algorithm: str = 'DQN',
                lstm_path: str = None,
                loss_method: str = None,
                from_checkpoint: dict = None,
                data_path: str ='./../data',) -> None:
        """
        Receive arguments and initialise the  class params.
        Parameters
        ----------
        """
        self.env = env 
        self.agent = agent
        self.data_path = data_path
        self.memory = deque(maxlen=max(batch_size+1, 1)) # Save Experience for policy update

        # States / Observation
        self.observation_space = observation_space
        self.state_size = observation_space[0]
        self.window_size = observation_space[1]
        
        # Action
        self.action_space = action_space

        # Train params
        self.batch_size = batch_size
        self.x_train_shape = (self.batch_size, self.window_size, self.state_size)
        self.y_train_shape = (self.batch_size, action_space)
        self.epoch = epoch
        self.gamma = gamma # Decay Constant for DQN
        self.init_episode = 1
        load_model = None

        # Logging params
        time_str=datetime.now().strftime('%Y%m%d_%H%M%S')
        self.train_folder=os.path.abspath(os.path.join(self.data_path, 
                time_str, algorithm))
        if isinstance(from_checkpoint, dict):
            if ('train_path' in from_checkpoint.keys()):
                self.train_folder=from_checkpoint['train_path']
                self.init_episode = from_checkpoint.get('init_episode', 1)
                load_model = from_checkpoint.get('load_model', None)

        self.train_log_dict = self.init_logging_dict()
        # self.train_log_dataframe = pd.DataFrame(columns=self.log_cols)

        # Init env controllable params
        self.env._update_log_folder(os.path.abspath(os.path.join(self.train_folder, 'episodes')))
        
        # Init agent controllable params
        # self.agent.build_model() # INIT MODEL
        if isinstance(load_model, str) and os.path.exists(load_model):
            self.agent.load_model(load_model,
                                learning_rate=learning_rate,
                                use_softmax=True,) 
        else:
            self.agent.build_model_LSTM(learning_rate=learning_rate,
                                lstm_path=lstm_path,
                                use_softmax=True,) 
        tf.compat.v1.get_default_graph().finalize()

    def rollout(self, n_episodes, run_per_episode):
        """
        Main function to start training. 
        Template:

        while data no over:
            state = init_state
            while episode not done:
                act = self.agent.compute_ation(state)
                state, reward = self.env.step(act)
                fit_data.append(state, action, reward)
                if batch_size reached:
                    self.batch_train(fit_data)
                if done or checkpoint:
                    self.save_data()

        Notes
        -----
        As in rl_agent.AI_Trader.train
        """
        train_cnt = 0
        total_profit = 0
        start_time = time.time()
        # debug_action = []
        # debug_dqnaction = []

        # Loop over every episode
        # for episode in range(1):
        for episode in range(self.init_episode, n_episodes + 1):
            print("Episode: {}/{}".format(episode, n_episodes))
            if episode % 10 == 0: # Increase Epsilon every 10 episodes
                self.agent.update_epsilon(increase_epsilon=0.5)
                print(f'on Episode {episode} set Eplison to {self.agent.epsilon} to find global minimum')
            self.env.reset(resample_data=True)
            run_profit = 0.0 # Init Profit on episode
            # Loop inside one episode over number runs 
            # for run in range(1):
            for run in range(1,run_per_episode+1):
                print("Episode: {}/{} || Run {}/{}".format(episode, 
                            n_episodes,run,run_per_episode))
                if run % 5 == 0: # Increase epsilon every 5 runs
                    self.agent.update_epsilon(increase_epsilon=0.25)
                    print(f'on Run {run} set Eplison to {self.agent.epsilon} to find global minimum')
                train_data={}
                run_profit = 0.0
                self.env.reset(resample_data=False)
                data_samples = self.env.episode_length
                old_action = np.array([0.0])
                state, _, _ = self.env.step(np.array([0.0]))
                for t in tqdm(range(data_samples)):
                    # Compute Action
                    tmp_wallet_value = self.env.wallet_value[0]
                    action = self.agent.compute_action(state)
                    # Transform Action from Policy to Env Requirement 
                    dqn_action = self.transform_to_dqn_action(action, old_action)
                    # debug_action.append(action)
                    # debug_dqnaction.append(dqn_action)
                    # Compute new step
                    next_state, reward, done = self.env.step(action=dqn_action)
                    # save Experience to Memory
                    self.memory.append((state, action, reward, next_state, done))
                    old_action = dqn_action
                    state = next_state
                    step_profit = self.env.wallet_value[0] - tmp_wallet_value
                    run_profit += step_profit

                    # save to logging
                    elapsed_time = time.time() - start_time
                    self.log_training(episode, run, action, state, reward, done, self.agent.epsilon, run_profit, elapsed_time, dqn_action)
                    # Check if is Done
                    if done:
                        self.env.log_episode_to_file(episode=episode, run=run)
                        break

                    # Train Policy if batch reached
                    if len(self.memory) > self.batch_size:
                        res = self.batch_train(self.memory)
                        self.memory.clear()
                        key_string=f'Epi_{episode}'
                        if key_string not in train_data:
                            train_data.update({key_string:{'loss':[res.history['loss']],'#trains':train_cnt,'epsilon':[self.agent.epsilon]}})
                        else:
                            train_data[key_string]['loss'].append(res.history['loss'])
                            train_data[key_string]['#trains']=train_cnt
                            train_data[key_string]['epsilon'].append(self.agent.epsilon)
                        train_cnt+=1

                    # Checkpoint data
                    if t >=100 and t % 100 == 0:
                        self.save_data(episode,train_data,save_model=False)
                        # Log Checkpoint Info to Screen
                        print(f'episode {episode}, run ({run}/{run_per_episode}) sample ({t}/{data_samples}).Profit {run_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)}')
                    # End Loop over one episode run
                
                self.save_data(episode,train_data,save_model=False)
                
                keras.backend.clear_session()
                # tf.reset_default_graph()
                gc.collect()
                # Log Run Info to Screen
                print(f'episode {episode}, finished run ({run}/{run_per_episode}). Run Profit {run_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)} ')
                # End Loop over all runs 

            # self.save_data(episode,train_data,save_model=True)
            # Log Episode Info to Screen
            total_profit+=run_profit
            print(f'episode {episode}/{n_episodes}. Profit {total_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)}')

            self.save_data(episode,train_data,save_model=True)

        # End Loop over episodes
        # return debug_action, debug_dqnaction

    def init_logging_dict(self) -> dict:
        self.log_cols=['episode', 'run', 'action', 'dqn_action', 'state', 
                    'reward', 'done','epsilon', 'profit', 'time_elapsed']
        tmp =  { key : [] for key in self.log_cols }
        return tmp
 
    def log_training(self, episode_log, 
                    run_log, action_log, state_log, reward_log, 
                    done_log, epsilon_log, profit_log, 
                    time_elapsed_log, dqn_act_log):
        """
        Add params to log dict
        """
        self.train_log_dict['episode'].append(episode_log)
        self.train_log_dict['run'].append(run_log)
        self.train_log_dict['action'].append(action_log)
        self.train_log_dict['dqn_action'].append(dqn_act_log)
        self.train_log_dict['state'].append(state_log)
        self.train_log_dict['reward'].append(reward_log)
        self.train_log_dict['done'].append(done_log)
        self.train_log_dict['epsilon'].append(epsilon_log)
        self.train_log_dict['profit'].append(profit_log)
        self.train_log_dict['time_elapsed'].append(time_elapsed_log)

    def transform_to_dqn_action(self, actions, saved_action):
        """
        """
        act_eval = np.argmax(actions)
        if act_eval == 1:
            act = 0.5
        elif act_eval == 2:
            act = 1.0
        elif act_eval == 3:
            act = 0.0
        else:
            # act = self.env.long_position[0]
            return np.array(saved_action).reshape((1,))
        return np.array([act]).reshape((1,))

    def batch_train(self, batch_memory):
        """
        Conduct batch train.

        Notes
        -----
        This function might incorporate the RL-Algo, 
        therefore will be different for different methods,
        as for example: DRL vs DQN

        Following DQN Algorithm the reward decay (gamma) is used in this function to define future rewards.

        As in rl_agent.AI_Trader.batch_train
        Returns
        -------
        result : 
            Tensorflow Training History
        """
        batch = []
        for i in range(max([len(batch_memory) - self.batch_size, 0]), len(batch_memory)):
            batch.append(batch_memory[i])
        # self.memory.clear()
        # init batch train vars for data
        x_train = np.zeros(self.x_train_shape)
        y_train = np.zeros(self.y_train_shape)

        # init state, action, reward for training
        state_tr, _, reward_tr, next_state_tr, done_tr = batch[0]
        state_input_tr = tf.convert_to_tensor(state_tr, dtype=tf.float32)
        action_tr = self.agent.model(state_input_tr, training=False)
        action_tr = action_tr.numpy()
        for index in range(1,len(batch)):
            if np.any(np.isnan(state_tr)) or \
                np.any(np.isnan(reward_tr)) or np.any(np.isnan(action_tr)):
                raise ValueError("nan value found")

            # Compute Reward Decay for DQN            
            state_input_tr = tf.convert_to_tensor(next_state_tr, dtype=tf.float32)
            action_next_tr = self.agent.model(state_input_tr, training=False)
            action_next_tr = action_next_tr.numpy()
            if not done_tr:
                reward_tr += self.gamma * np.max(action_next_tr)

            # Compute new target
            target = action_tr
            id_act = np.argmax(target)
            target[0,id_act] = reward_tr

            # Update training data
            x_train[index]= state_tr[0]
            y_train[index]= target[0]
            # update state, action, reward for training
            state_tr, _, reward_tr, next_state_tr, done_tr = batch[index]
            action_tr = action_next_tr

        # Batch Train
        
        # x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        # y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        gc.collect()
        result=self.agent.model.fit(x_train, y_train, 
                epochs=self.epoch, 
                verbose=0,)
                # callbacks=[CustomCallback()])
        self.agent.update_epsilon()
        return result

    def save_data(self,episode,train_data,save_model=True):
        """
        Save data from rollout, if changed to save data per episode, then move this function to env
        
        Notes
        -----
        As in rl_agent.AI_Trader.save_data
        """
        # make folder if doesnt exists
        time_str=datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.train_folder,exist_ok=True)
        
        # save train data and model
        with open(self.train_folder+"/train_data.json","w") as out_file:
            json.dump(train_data,out_file)
        if save_model:
            self.agent.model.save(self.train_folder+"models_ai_trade_{}_{}.h5".format(time_str,episode))
        
        df_path = self.train_folder + f"/Trainer_Data_{episode}.csv"
        train_log_dataframe = pd.DataFrame(columns=self.log_cols)
        if os.path.exists(df_path):
            train_log_dataframe= pd.read_csv(df_path, sep=';')
        
        # Save info from checkpoint to train_log_dataframe
        tmp = pd.DataFrame.from_dict(self.train_log_dict)
        train_log_dataframe = pd.concat([train_log_dataframe, tmp])
        # Reinit log dict to avoid double logging
        self.train_log_dict = self.init_logging_dict() 
        # Save  train_log_dataframe to file
        train_log_dataframe.to_csv(df_path, sep=';', index=False)
        del train_log_dataframe
        print('Data saved')

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    obs_space = (8,20)
    act_space = 4

    money = 10000
    fee = 0.001
    episodes = 1
    runs_p_eps = 1

    env = BTCMarket_Env(observation_space = obs_space,
                action_space = act_space,
                start_money = money,
                trading_fee= fee)
    agent = Trader_Agent(observation_space = obs_space,
                action_space = act_space,
                epsilon = 0.01)
    dqntrainer = DQNTrainer(env, agent,
                observation_space = obs_space,
                action_space = act_space,
                batch_size=100,
                lstm_path = "./../notebooks/best_models/11_mar_2023/lstm_2.h5")

    dqntrainer.rollout(n_episodes=episodes, run_per_episode=runs_p_eps)

