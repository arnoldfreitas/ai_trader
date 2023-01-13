import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from tensorflow import keras
from typing import Tuple



class Trader_Agent():
    '''
    Trader class.

    Build model/load model, calculates action.

    '''
    def __init__(self, 
                observation_space: tuple, 
                action_space: tuple) -> None:
        """
        Receive arguments and initialise the  class params.
        Parameters
        ---------
        observation_space: tuple
            Shape of observation / state space
        action_space: tuple
            Shape of action space
        """
        self.model = None
        self.epsilon = 0.0
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
        model = keras.models.Sequential([        
            keras.Input(shape=(self.state_size,)),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=self.action_space, activation='linear')
            ])
        #TODO: Make learning-rate changeable.
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
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

    def compute_action(self, state: np.ndarray) ->  np.ndarray:
        """
        Performs Trade action from an given state. Uses epsilon greedy method or model.predict to define action.
        
        Note
        ----
        As in rl_agent.AI_Trader.trade

        Returns
        -------
            Computed actions
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
      
        actions = self.model.predict(tf.reshape(tf.convert_to_tensor(state[0],dtype=np.float32),shape=(1,self.state_size)),verbose = 0)
        return np.argmax(actions)