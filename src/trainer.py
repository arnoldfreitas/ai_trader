import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from tensorflow import keras
from env import BTCMarket_Env
from agent import Trader_Agent


class Trainer():
    '''
    Trainer class

    Reinforcement Algorithm. Trains the agent (Policy Network).
    '''
    def __init__(self, env, agent) -> None:
        """
        Receive arguments and initialise the  class params.
        Parameters
        ----------
        """
        self.env = None 
        self.agent = None
        pass

    def rollout(self):
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
        pass
    
    def batch_train(self):
        """
        Conduct batch train.

        Notes
        -----
        This function might incorporate the RL-Algo, 
        therefore will be different for different methods,
        as for example: DRL vs DQN 

        As in rl_agent.AI_Trader.batch_train
        """
        pass

    def save_data(self):
        """
        Save data from rollout, if changed to save data per episode, then move this function to env
        
        Notes
        -----
        As in rl_agent.AI_Trader.save_data
        """
        pass

    def dataset_loader(self):
        """
        Load dataset that will be used in the rollout.

        Notes
        -----
        As in rl_agent.AI_Trader.dataset_loader
        """
        pass

    def stock_price_format(self,n):
        '''
        Reformat Stock Price Value to fit string standard format

        Returns
        -------
        string : str
            Formated price stock
        '''
        if n < 0:
            return "- # {0:2f}".format(abs(n))
        else:
            return "$ {0:2f}".format(abs(n))

    @staticmethod 
    def loadData(onefile=False,asset=None):
        out=[]
        for file in os.listdir('./data'):
            if asset is not None and asset not in file:
                continue
            if 'histData_dt1800.0s' in file:
                out.append(pd.read_csv('./data/'+file))
        if onefile:
            out_df=pd.DataFrame(columns=out[0].columns)
            for item in out:
                out_df=pd.concat([out_df,item])
            return out_df
        else:
            return out