import random
import os
import traceback
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras



class Trader_Agent():
    '''
    Trader class.

    Build model/load model, calculates action.

    '''
    def __init__(self, 
                observation_space: tuple, 
                action_space: tuple,
                model_name: str ="AITrader",
                data_path: str ='./../data',
                load_model: str =None,
                epsilon: float = 1.0,
                epsilon_final: float = 0.01,
                epsilon_decay: float = 0.995,) -> None:
        """
        Receive arguments and initialise the  class params.
        Parameters
        ---------
        observation_space: tuple
            Shape of observation / state space: (n_inputs, window_size)
        action_space: tuple
            Shape of action space: 
            DQN: (,4) (reward, act_buy50, act_buy100, act_sell)
            DRL: (,2) (reward, act_continuos)
        """
        self.model = None
        self.data_path=data_path

        # States / Observation
        self.observation_space = observation_space
        self.state_size = observation_space[0]
        self.window_size=observation_space[1]
        
        # Action
        self.action_space = action_space
        
        # Exploration Params
        self.epsilon = epsilon # exploration or not 1 is full random, start with 1 
        self.epsilon_final = epsilon_final # final epsilon
        self.epsilon_decay = epsilon_decay 

        # Model Params. Build/Load Model
        self.model_name = model_name
        if (load_model is not None) and (isinstance(load_model, str)):
            try:
                self.model = self.load_model()
            except:
                print('Model loading failed:')
                print(traceback.print_exc())
                self.model=self.model_builder()
        else:
            self.model=self.model_builder()

    def build_model(self, learning_rate=1e-3):
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
        model.compile(loss='mse', 
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
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
        model = keras.models.load_model(
                self.data_path+"/Bot/models/ai_trade_{}_{}.h5".format(load_date,load_epi))

        print("model: ai_trade_{}_{} loaded. Eplison set to {}.".format(
                load_date,load_epi,self.epsilon))
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