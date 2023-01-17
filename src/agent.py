import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from env import BTCMarket_Env

class Trader_Agent():
    '''
    Trader class.

    Build model/load model, calculates action.

    '''
    
    def __init__(self, 
                observation_space: tuple, 
                action_space: tuple,
                learning_rate: float) -> None:
        """
        Receive arguments and initialise the  class params.
        Parameters
        ---------
        observation_space: tuple
            Shape of observation / state space
        action_space: tuple
            Shape of action space

            # Because of continuous action space it would make sense like: (dimension, [range]) = (1, [-1, 1])
            # action_space[0] = 1 = dimension, action_space[1] = [-1, 1] = range of output of policy network and for random exploring
            See TODO below reguarding the action space
        """
        self.model = None
        self.epsilon = 0.0

        #TODO: action_space as input important? 
        self.output_dim = action_space[0] ### Überhaupt wichtig hier action_space zu übergeben? Ist eigentlich immer derselbe
        self.output_range = action_space[1] ### Same here
        '''
        Alternativ:

        self.output_dim = 1
        self.output_range = [-1,1]
        '''
        self.learning_rate = learning_rate
        self.state_size = observation_space
        pass
    
    def build_model(self):
        """
        Build Policy model with predefined architecture.

        Notes
        -----
        As in rl_agent.AI_Trader.model_builder

        Returns
        -------
            Tensorflow Model 
        """

        ### Code from rl_agent
        '''model = keras.models.Sequential([        
            keras.Input(shape=(self.state_size,)),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=self.action_space, activation='linear')
            ])
        #TODO: Make learning-rate changeable.
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model'''


        ### Continuous action space with MLP (just for starting purposes)
        model = keras.models.Sequential([        
            keras.Input(shape=(self.state_size,)),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=self.output_dim, activation='tanh')
            ])

        #TODO: Build RNN (LSTM) as policy network

        #TODO: Design Utility Function in BTCMarket_Env
        # Loss function has to be negative in order to perform gradient ascent
        custom_loss_func = -BTCMarket_Env.compute_utility 

        model.compile(loss=custom_loss_func, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learing_rate))

        return model
    
    def load_model(self, model_path: str):
        """
        Load TensorFlow Model from h5 file. 

        Note
        ----
        Set self.epsilon to 0.5
        As in rl_agent.AI_Trader.load_model

        Returns
        -------
            Tensorflow Model 
        """
        epi_list=[]
        date_list=[] 
        for file in os.listdir(self.data_path+"/Bot/models"):
            if '.h5' in file:
                date_list.append(file.split('.')[0].split('_')[2])
                epi_list.append(int(file.split('.')[0].split('_')[3]))
        load_epi=max(epi_list)
        load_date=date_list[epi_list.index(load_epi)] 
        # TODO fix load model line
        model = keras.models.load_model(self.data_path+"/Bot/models/ai_trade_{}_{}.h5".format(load_date,load_epi))
        self.epsilon=0.5
        print("model: ai_trade_{}_{} loaded. Eplison set to {}.".format(load_date,load_epi,self.epsilon))
        return model

    def compute_action(self, state: np.ndarray) ->  float:
        """
        Performs Trade action from a given state. Uses epsilon greedy method or model.predict to define action.
        
        Note
        ----
        Old implementaion of this method: As in rl_agent.AI_Trader.trade

        Idea:
        [0.2, 1] --> buy(/long) with 0.x % of total money
        ]-0.2, 0.2[ --> hold/wait
        [-1, -0.2] --> sell(/short) 0.x % of total money

        #TODO: Probably build an extra function for this in the BTCMarket_Env Class

        Returns
        -------
            Computed action
        """
        #TODO: Is this the way to go? Or everytime compute action and sometimes add noise to the computed action?
        if random.random() <= self.epsilon:
            return random.uniform(*self.output_range)
      
        # round computed value to one decimal point
        action_val = round(self.model.predict(tf.reshape(tf.convert_to_tensor(state[0],dtype=np.float32),shape=(1,self.state_size)),verbose = 0), 1)
        # Idea is to make action_space smaller
        if abs(action_val) < 0.2:
            return 0
        return action_val