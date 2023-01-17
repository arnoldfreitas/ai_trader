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