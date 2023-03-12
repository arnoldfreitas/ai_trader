import os
from datetime import datetime
import json
import shutil
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm_notebook, tqdm
from matplotlib import pyplot as plt

from env import BTCMarket_Env
from agent import Trader_Agent
from collections import deque


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
                algorithm: str = 'DQN',
                lstm_path: str = None,
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

        # Logging params
        time_str=datetime.now().strftime('%Y%m%d_%H%M%S')
        self.train_folder=os.path.abspath(os.path.join(self.data_path, 
                    time_str, algorithm))
        self.train_log_dict = self.init_logging_dict()
        self.train_log_dataframe = pd.DataFrame(columns=self.log_cols)

        # Init env controllable params
        self.env._update_log_folder(self.train_folder)
        
        # Init agent controllable params
        # self.agent.build_model() # INIT MODEL

        self.agent.build_model_LSTM(lstm_path=lstm_path) 
        

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
        # Loop over every episode
        # for episode in range(1):
        for episode in range(1, n_episodes + 1):
            print("Episode: {}/{}".format(episode, n_episodes))
            if episode % 10 == 0: # Increase Epsilon every 10 episodes
                self.agent.update_epsilon(increase_epsilon=0.5)
                print(f'on Episode {episode} set Eplison to {self.agent.epsilon} to find global minimum')
            run_profit=0.0 # Init Profit on episode
            # Loop inside one episode over number runs 
            # for run in range(1):
            for run in range(1,run_per_episode+1):
                print("Episode: {}/{} || Run {}/{}".format(episode, 
                            n_episodes,run,run_per_episode))
                if run % 5 == 0: # Increase epsilon every 5 runs
                    self.agent.update_epsilon(increase_epsilon=0.5 -(run/run_per_episode)*0)
                    print(f'on Run {run} set Eplison to {self.agent.epsilon} to find global minimum')
                train_data={}
                run_profit = 0
                self.env.reset()
                data_samples = self.env.episode_length
                state, _, _ = self.env.step(np.array([0]))
                for t in tqdm(range(data_samples)):
                    # Compute Action
                    tmp_wallet_value = env.wallet_value
                    action = self.agent.compute_action(state)
                    # Transform Action from Policy to Env Requirement 
                    dqn_action = self.transform_to_dqn_action(action)
                    # Compute new step
                    next_state, reward, done = self.env.step(action=dqn_action)
                    # save Experience to Memory
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    step_profit = env.wallet_value - tmp_wallet_value
                    run_profit += step_profit
                    # save to logging
                    self.log_training(episode, run, action, state, reward, done, self.agent.epsilon)
                    # Check if is Done
                    if done:
                        env.log_episode_to_file(episode=episode, run=run)    
                        break

                    # Train Policy if batch reached
                    if len(self.memory) > self.batch_size:
                        res = self.batch_train()
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
                        print(f'episode {episode}, run ({run}/{run_per_episode}) sample ({t}/{data_samples}).Profit {run_profit}')
                
                # Log Run Info to Screen
                print(f'episode {episode}, finished run ({run}/{run_per_episode}). Run Profit {run_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)}')
            
            # Log Episode Info to Screen
            total_profit+=run_profit
            print(f'episode {episode}/{episodes}. Profit {total_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)}')

            self.save_data(episode,train_data,save_model=True)

    def init_logging_dict(self) -> dict:
        self.log_cols=['episode', 'run', 'action', 'state', 
                    'reward', 'done','epsilon']
        return dict.fromkeys(self.log_cols, [])
 
    def log_training(self, episode, run, action, state, reward, done, 
                epsilon):
        """
        Add params to log dict
        """
        self.train_log_dict['episode'].append(episode)
        self.train_log_dict['run'].append(run)
        self.train_log_dict['action'].append(action)
        self.train_log_dict['state'].append(state)
        self.train_log_dict['reward'].append(reward)
        self.train_log_dict['done'].append(done)
        self.train_log_dict['epsilon'].append(epsilon)

    def transform_to_dqn_action(self, actions):
        """
        """
        act_eval = np.argmax(actions)
        if act_eval == 1:
            action = 0.5
        elif act_eval == 2:
            action = 1.0
        elif act_eval == 3:
            action = 0.0
        else:
            action = self.env.long_position
        return np.array([action])

    def batch_train(self):
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
        for i in range(max([len(self.memory) - self.batch_size, 0]), len(self.memory)):
            batch.append(self.memory[i])
        self.memory.clear()
        # init batch train vars for data
        x_train = np.zeros(self.x_train_shape)
        y_train = np.zeros(self.y_train_shape)

        # init state, action, reward for training
        state, _, reward, next_state , done = batch[0]
        action = self.agent.model.predict(state,verbose = 0)
        for index in range(1,len(batch)):
            # Unused code for keeping track of the past
            # state = tf.reshape(tf.convert_to_tensor(state,dtype=np.float32),
            #                     shape=(1,self.state_size*self.window_size))
            # next_state = tf.reshape(tf.convert_to_tensor(next_state,dtype=np.float32),
            #                     shape=(1,self.state_size*self.window_size))
            # check for nan values, or may occur errors during training
            if np.any(np.isnan(state)) or \
                np.any(np.isnan(reward)) or np.any(np.isnan(action)):
                raise ValueError("nan value found")

            # Compute Reward Decay for DQN
            action_next = self.agent.model.predict(next_state,verbose = 0)
            if not done:
                reward += self.gamma * np.max(action_next)

            # Compute new target
            target = action
            id_act = np.argmax(target)
            target[0,id_act] = reward

            # Update training data
            x_train[index]= state[0]
            y_train[index]= target[0]
            # update state, action, reward for training
            state, _, reward, next_state , done = batch[index]
            action = action_next

        # Batch Train
        result=self.agent.model.fit(x_train, y_train, 
                epochs=self.epoch, verbose=1)
        agent.update_epsilon()
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
        
        # Save info from checkpoint to train_log_dataframe
        tmp = pd.DataFrame.from_dict(self.train_log_dict)
        self.train_log_dataframe = pd.concat([self.train_log_dataframe, tmp])
        # Reinit log dict to avoid double logging
        self.train_log_dict = self.init_logging_dict() 
        # Save  train_log_dataframe to file
        self.train_log_dataframe.to_csv(self.train_folder + f"/Trainer_Data.csv")
        print('Data saved')

if __name__ == "__main__":
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

