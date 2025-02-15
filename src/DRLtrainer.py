import os
from datetime import datetime
import json
import shutil
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# tf.compat.v1.disable_eager_execution()
import gc
from tqdm import tqdm_notebook, tqdm
from matplotlib import pyplot as plt
import time

from env import BTCMarket_Env
from agent import Trader_Agent
from collections import deque

np.random.seed(42)
random.seed(42)

class DRLLossFunctions(keras.losses.Loss):
    def __init__(self, loss_method="stantard_loss"):
        super().__init__()
        self.name="DRL Custom Loss"
        self.environment_data = None
        self.loss_str = loss_method
        if hasattr(self, loss_method):
            self.loss_fn = getattr(self, loss_method)
        else:
            self.loss_fn = self.stantard_loss

    def update_environment_data(self, data):
        self.environment_data = data.copy()

    def reward_differential_sharpe_ratio(self, 
                y_true,
                y_pred,) -> float:
        """
        Function to compute Differential Sharpe Ratio as loss function.

        Notes
        -----
        build state as in rl_agent.AI_Trader.get_reward_money

        Parameters
        ----------
        y_true := time_step in self.environment_data
        y_pred := actions

        Returns
        -------
        loss
        """
        # tau_decay parameter 
        tau_decay = 0.99
        # Gamma is set to this value in the paper (rrl agent)
        gamma = 0.001
        # get historical data for calculations
        price_change = tf.gather(self.environment_data["price_change"], tf.cast(y_true, tf.int32))

        # idx = tf.roll(tf.cast(y_true, tf.int32), shift=-1, axis=1)
        idx = tf.clip_by_value(tf.math.add(tf.cast(y_true, tf.int32),-1), 0, 1000)
        # old_price = tf.gather(self.environment_data["product_price"], 
        #             idx)
        var_old = tf.gather(self.environment_data["var_returns_squared"], 
                    idx)
        exp_return_old = tf.gather(self.environment_data["expected_return"], 
                    idx)
        # Price Change
        # price_change = tf.math.divide_no_nan((actual_price - old_price), old_price)
        # price_change = actual_price - old_price
        # Return on action
        representative_return = tf.math.multiply(y_pred, price_change) 
        # Reutrns standard deviation
        variance_returns_squared = tf.math.multiply(tau_decay, var_old) + tf.math.multiply((1 - tau_decay), tf.math.square(representative_return - exp_return_old))
        # Update expected reward
        expected_return = tau_decay * exp_return_old + (1 - tau_decay)* representative_return
        # Calculate loss
        dsr = expected_return - (gamma/ 2) * variance_returns_squared
        loss = -tf.reduce_mean(dsr)
        return loss


    def stantard_loss(self,  y_true, y_pred):
        '''
        Function to compute standard loss for gradient ascent in DRL Algorithm.
        
        Notes
        -----
        Function might generates very high gradients, therefore we should use low learning rates
        
        Parameters
        ----------
        y_true := rewards
        y_pred := actions
        '''
        tmp = tf.reduce_mean(y_true*y_pred)
        loss = tf.math.scalar_mul(-1, tmp, name=None)
        return loss


    def call(self, y_true, y_pred):
        '''
        Function to compute loss for DRl
        Parameters
        ----------
        y_true := 
        y_pred := actions
        '''
        loss = self.loss_fn(y_true, y_pred)
        return loss

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs=None):
        # Housekeeping
        gc.collect()
        keras.backend.clear_session()

class DRLTrainer():
    '''
    Trainer class

    Reinforcement Algorithm. Trains the agent (Policy Network).
    '''
    def __init__(self, env, agent,
                observation_space: tuple, 
                action_space: int,
                batch_size: int,
                epoch: int = 1,
                gamma: float = 0.95,
                learning_rate: float =1e-3,
                algorithm: str = 'DRL',
                lstm_path: str = None,
                loss_method: str = "stantard_loss",
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

        self.memory = deque(maxlen=max([batch_size+1,1])) # Save Experience for policy update
        # States / Observation
        self.observation_space = observation_space
        self.state_size = observation_space[0]
        self.window_size = observation_space[1]
        
        # Action - int?
        self.action_space = action_space

        # Train params
        self.batch_size = batch_size
        self.x_train_shape = (self.batch_size, self.window_size, self.state_size)
        self.y_train_shape = (self.batch_size, action_space)
        self.epoch = epoch
        self.gamma = gamma # Decay Constant for DQN
        self.init_episode = 1

        # Logging params
        load_model = None
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
        
        # custom_loss = DRLLossFunctions()
        self.custom_loss = DRLLossFunctions(loss_method=loss_method)
        self.use_standard_loss = loss_method == "stantard_loss"

        # Init agent controllable params and init model
        if isinstance(load_model, str) and os.path.exists(load_model):
            self.agent.load_model(load_model, 
                                learning_rate=learning_rate,
                               loss_function=self.custom_loss,) 
        else:
            self.agent.build_model_LSTM(learning_rate=learning_rate,
                               loss_function=self.custom_loss,
                               lstm_path=lstm_path) 

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
        # Loop over every episode
        for episode in range(self.init_episode, n_episodes + 1):
            print("Episode: {}/{}".format(episode, n_episodes))
            if episode % 10 == 0: # Increase Epsilon every 10 episodes
                self.agent.update_epsilon(increase_epsilon=0.5)
                print(f'on Episode {episode} set Eplison to {self.agent.epsilon} to find global minimum')
            self.env.reset(resample_data=True)
            run_profit=0.0 # Init Profit on episode
            # Loop inside one episode over number runs 
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
                state, _, _ = self.env.step(np.array([0]))
                for t in tqdm(range(data_samples)):
                # for t in tqdm(range(100)):
                    # Compute Action
                    tmp_wallet_value = self.env.wallet_value[0]
                    action = self.agent.compute_action(state)
                    # debug_action.append(action)
                    # round action to one decimal point (that we dont take to small actions)
                    rounded_action = np.round(action, 1) ####### we also round the action in the step() function, just leave both for extra safety.
                    # Compute new step
                    next_state, reward, done = self.env.step(action=rounded_action)
                    # save Experience to Memory
                    if not self.use_standard_loss:
                        self.memory.append((state, action, t, t+1, done))
                    else:
                        self.memory.append((state, action, reward, next_state, done))

                    state = next_state
                    step_profit = self.env.wallet_value[0] - tmp_wallet_value
                    run_profit += step_profit

                    # save to logging
                    elapsed_time = time.time() - start_time
                    self.log_training(episode, run, action, state, reward, done, self.agent.epsilon, run_profit, elapsed_time)
                    # Check if is Done
                    if done:
                        self.env.log_episode_to_file(episode=episode, run=run)    
                        break

                    # Train Policy if batch reached
                    if len(self.memory) > self.batch_size:
                        if not self.use_standard_loss:
                            self.custom_loss.update_environment_data(self.env.reward_log_dict)
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
                        print(f'episode {episode}, run ({run}/{run_per_episode}) sample ({t}/{data_samples}).Profit {run_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)}')
                    # End Loop over one episode run
                
                self.save_data(episode,train_data,save_model=False)
                
                keras.backend.clear_session()
                # tf.reset_default_graph()
                gc.collect()
                # Log Run Info to Screen
                print(f'episode {episode}, finished run ({run}/{run_per_episode}). Run Profit {run_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)}')
                # End Loop over all runs 

            # Log Episode Info to Screen
            total_profit+=run_profit
            print(f'episode {episode}/{n_episodes}. Profit {total_profit} || money available: {(self.env.money_available)},  wallet value: {(self.env.wallet_value)}')

            self.save_data(episode,train_data,save_model=True)

        # End Loop over episodes
        # return debug_action

    def init_logging_dict(self) -> dict:
        self.log_cols=['episode', 'run', 'action', 'state', 
                    'reward', 'done','epsilon', 'profit', 'time_elapsed']
        tmp =  { key : [] for key in self.log_cols }
        return tmp
 
    def log_training(self, episode_log, run_log, 
                action_log, state_log, reward_log, 
                done_log, epsilon_log, profit_log, 
                time_elapsed_log):
        """
        Add params to log dict
        """
        self.train_log_dict['episode'].append(episode_log)
        self.train_log_dict['run'].append(run_log)
        self.train_log_dict['action'].append(action_log)
        self.train_log_dict['state'].append(state_log)
        self.train_log_dict['reward'].append(reward_log)
        self.train_log_dict['done'].append(done_log)
        self.train_log_dict['epsilon'].append(epsilon_log)
        self.train_log_dict['profit'].append(profit_log)
        self.train_log_dict['time_elapsed'].append(time_elapsed_log)

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
        y_target = np.zeros(self.y_train_shape)
            
        # For DRL, we just take all saved values from the env for training
        # Compute Gradients of the Reward to all weights and update
        for index in range(0,len(batch)):
            state, _, reward, _, _= batch[index]
            # check for nan values, or may occur errors during training
            #if np.any(np.isnan(state)) or \
            #    np.any(np.isnan(reward)) or np.any(np.isnan(action)):
            #    raise ValueError("nan value found")

            # we dont need the action for training we just declare it here to give some y_pred to keras because it needs it.
            # Our custom loss function just need y_target namely the reward
            # y_pred = action
            x_train[index] = state[0]
            y_target[index,0] = reward

        # Batch Train (in this case on-line traing without batches) 
        # we can just set the batch to 1 and it will do online training
        # x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        # y_target = tf.convert_to_tensor(y_target, dtype=tf.float32)
        result=self.agent.model.fit(x_train, y_target, 
                epochs=self.epoch, 
                verbose=1,)
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
    act_space = 1
    action_domain = (-1.0,1.0) # (0.0, 1.0)

    money = 10000
    fee = 0.001
    episodes = 1
    runs_p_eps = 1

    env = BTCMarket_Env(observation_space = obs_space,
                action_space = act_space,
                reward_function="reward_differential_sharpe_ratio",
                start_money = money,
                trading_fee= fee)
    agent = Trader_Agent(observation_space = obs_space,
                action_space = act_space,
                action_domain = action_domain,
                epsilon = 0.1)
    drltrainer = DRLTrainer(env, agent,
                observation_space = obs_space,
                action_space = act_space,
                batch_size=50,
                lstm_path = "./../notebooks/best_models/11_mar_2023/best_model_sequential_20back_10ahead.h5") # best_model_sequential_20back_10ahead lstm_2

    drltrainer.rollout(n_episodes=episodes, run_per_episode=runs_p_eps)

