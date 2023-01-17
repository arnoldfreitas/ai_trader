import os
import datetime
import json
import shutil
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm_notebook, tqdm

from env import BTCMarket_Env
from agent import Trader_Agent


class Trainer():
    '''
    Trainer class

    Reinforcement Algorithm. Trains the agent (Policy Network).
    '''
    def __init__(self, env, agent,
                observation_space: tuple, 
                action_space: tuple,
                batch_size: int,
                gamma: float = 0.95,
                data_path: str ='./../data',) -> None:
        """
        Receive arguments and initialise the  class params.
        Parameters
        ----------
        """
        self.env = env 
        self.agent = agent
        self.data_path = data_path
        
        self.gamma = gamma # Decay Constant for DQN
        self.batch_size = batch_size
        self.memory = None
        # States / Observation
        self.observation_space = observation_space
        self.state_size = observation_space[0]
        self.window_size = observation_space[1]
        
        # Action
        self.action_space = action_space
        

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
        self.epi_cols=['episode','#buy_actions','#sell_actions','money','fee','profit','epsilon']
        #action_data=pd.DataFrame(columns=['episode','run','date','action','state','money_free','money_fiktiv','invest','fee','reward','profit'])
        epi_dataFrame=pd.DataFrame(columns=self.epi_cols)

        # Loop over every episode
        # for episode in range(1):
        for episode in range(1, n_episodes + 1):
            print("Episode: {}/{}".format(episode, n_episodes))
            if episode % 10 == 0: # Increase Epsilon every 10 episodes
                self.agent.update_epsilon(increase_epsilon=0.5)
                print(f'on Episode {episode} set Eplison to {self.agent.epsilon} to find global minimum')
            run_profit=0.0 # Init Profit on episode
            # Init dataframe for runs
            action_data=pd.DataFrame(columns=['episode','run','timestep','date','action','state','money_free','money_fiktiv','invest','fee','reward','profit'])
            # Loop inside one episode over number runs 
            # for run in range(1):
            for run in range(1,run_per_episode+1):
                print("Episode: {}/{} || Run {}/{}".format(episode, 
                            n_episodes,run,run_per_episode))
                if run % 5 == 0: # Increase epsilon every 5 runs
                    self.agent.update_epsilon(increase_epsilon=0.5 -(run/run_per_episode)*0)
                    print(f'on Run {run} set Eplison to {self.agent.epsilon} to find global minimum')
                train_data={}
                self.env.reset()
                data_samples = self.env.episode_length
                state = self.env.step(np.zeros(self.action_space),0)
                for t in tqdm(range(data_samples)):
                    action = self.agent.compute_action(state)
                    dqn_action, btc_wallet_change = self.transforme_to_dqn_action(action)
                    next_state, reward, done = self.env.step(action=action, 
                                                btc_wallet_variaton=btc_wallet_change)

                    state = next_state

    def transforme_to_dqn_action(self, actions):
        """
        """
        act = np.argmax(actions)
        if act == 1:
            btc_change = 0.5
        elif act == 2:
            btc_change = 1
        elif act == 3:
            btc_change = -1
        else:
            btc_change = 0

        return act, btc_change
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
        # for i in range(1):
        for i in range(len(self.memory) - self.batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        # Change here to fit new RL-Algo
        id = 0
        for state, action, reward, next_state, done in batch:
            reward = reward
            # Comput Reward Decay for DQN
            if not done:
                reward += self.gamma * np.amax(self.agent.model.predict(next_state,verbose = 0)[0]) 
                # reward += self.gamma * np.amax(batch[id+1,1][0]) 

            target = self.agent.model.predict(state,verbose = 0)
            target[0][action] = reward

            result=self.agent.model.fit(state, target, epochs=5, verbose=0)
            id +=1

        self.agent.update_epsilon()
        
        return result

    def save_data(self,
                action_data,epi_data,episode,train_data,
                epi_dataFrame,overwrite=False,save_model=True):
        """
        Save data from rollout, if changed to save data per episode, then move this function to env
        
        Notes
        -----
        As in rl_agent.AI_Trader.save_data
        """
        # TODO: Rewrite Function
        save_str=datetime.now().strftime('%Y%m%d')
        save_path=self.data_path+'/Bot/RL_Bot/'+save_str
        index=0
        if overwrite and os.path.exists(save_path):
            shutil.rmtree(save_path)
        #while os.path.exists(save_path):
        #    save_path=save_path+'_'+str(index)
        os.makedirs(save_path,exist_ok=True)
        epi=action_data['episode'].unique()[0]
        action_data.to_csv(save_path+f"/action_data_e{epi}.csv")
        with open(save_path+"/train_data.json","w") as out_file:
            json.dump(train_data,out_file)
        if save_model:
            self.model.save(self.data_path+"/Bot/models/ai_trade_{}_{}.h5".format(save_str,episode))
        tmp=pd.DataFrame(epi_data,columns=self.epi_cols)
        epi_dataFrame=pd.concat([epi_dataFrame,tmp])
        epi_dataFrame.to_csv(save_path+"/Epi_data.csv")
        print('Data saved')