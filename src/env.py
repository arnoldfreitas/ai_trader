import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

class BTCMarket_Env():
    '''
    Environment Class.
    
    Interaction of the agent with the environment. Calculates utility/reward. Observes states and executes actions (env.step()).
    '''
    def __init__(self) -> None:
        """
        Receive arguments and initialise the  class params.
        """
        pass
    
    def reset(self) -> None:
        """
        Restart/Start episodes 
        
        Notes
        -----
        It is embedded inside rl_agent.AI_Trader.train, when a episode is over and another start
        """
        pass
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Receives an action (Output from Agent.compute_action) and computes the next observation/state and its reward.
        
        Notes
        -----
        build state as in rl_agent.AI_Trader.state_creator
        """
        pass

    def compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Function to compute reward based on state and action.

        Notes
        -----
        build state as in rl_agent.AI_Trader.get_reward_money
        """
        pass

    def sigmoid(self,x):
        try:
            result = math.exp(-x)
        except OverflowError:
            result = math.inf
        return 1 /(1 + result)

    @staticmethod 
    def getrandomSample(data,period):
        start=random.randint(0,len(data)-period)
        return data.iloc[start:start+period,:] 