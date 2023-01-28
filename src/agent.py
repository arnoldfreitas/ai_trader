import random
import os
import traceback
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from env import BTCMarket_Env

def loss_function(y_true, y_pred):
    '''
    Function to compute loss for gradient ascent.

    action is of shape (4)
    y_true = np.array([reward] * 4)

    y_true: we can compute, therefore we set as reward. Shape must be equal action shape.
    y_pred: actions from policy NN given through fit.
    '''
    loss = - y_true[0]

    return loss

class Trader_Agent():
    '''
    Trader class.

    Build model/load model, calculates action.

    '''
    
    def __init__(self, 
                observation_space: tuple, 
                action_space: tuple,
                action_range: list, # min min
                ###learning_rate: float) -> None:
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
        Shape of action space: 
        DQN: (,4) (act_hold, act_buy50, act_buy100, act_sell)
        DRL: (,2) (act_continuos)

        # Because of continuous action space it would make sense like: (dimension, [range]) = (1, [-1, 1])
        # action_space[0] = 1 = dimension, action_space[1] = [-1, 1] = range of output of policy network and for random exploringe

        #TODO: action_space as input important? 
        self.output_dim = action_space[0] ### Überhaupt wichtig hier action_space zu übergeben? Ist eigentlich immer derselbe
        self.output_range = action_space[1] ### Same here
        '''
        Alternativ:

        self.output_dim = 1
        self.output_range = [-1,1]
        '''
        self.learning_rate = learning_rate
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

        """
        We have an LSTM, which predict the next n time steps for BTC close value.

        model_nur_lestm  =self.load_LSTM(inputs, trainable=False)
        model_nur_lestm.add_leayer(
            inputs.append(model_nur_lest.outputs),
            layer1
            layer2
            layer3
            output_layer
        )
        """
        
        ### Continuous action space with MLP for BTC 0-1 (just for starting purposes)
        # 1: buy all, 0: sell all
        # for trading perpetual swap change activation function of outputlyer to "tanh"
        model = keras.models.Sequential([        
            keras.Input(shape=(self.state_size,)),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=self.action_space, activation='sigmoid')
            ])

        #TODO: Build RNN (LSTM) as policy network

        #TODO: Design Utility Function in BTCMarket_Env
        # Loss function has to be negative in order to perform gradient ascent
        custom_loss_func = loss_function 

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
        model = keras.models.load_model(
                self.data_path+"/Bot/models/ai_trade_{}_{}.h5".format(load_date,load_epi))

        print("model: ai_trade_{}_{} loaded. Eplison set to {}.".format(
                load_date,load_epi,self.epsilon))
        return model

    def compute_action(self, state: np.ndarray) ->  float:
        """
        Performs Trade action from a given state. Uses epsilon greedy method or model.predict to define action.
        
        Note
        ----
        Old implementaion of this method: As in rl_agent.AI_Trader.trade

        #TODO: Probably build an extra function for this in the BTCMarket_Env Class

        Returns
        -------
            Computed action
        """
        #TODO: Is this the way to go? Or everytime compute action and sometimes add noise to the computed action?
        if random.random() <= self.epsilon:
            return random.uniform(*self.output_range)
      
        # round computed value to one decimal point
        # action_val = round(self.model.predict(tf.reshape(tf.convert_to_tensor(state[0],dtype=np.float32),shape=(1,self.state_size)),verbose = 0), 1)
        action_val = self.model.predict(tf.reshape(tf.convert_to_tensor(state[0],dtype=np.float32),shape=(1,self.state_size)),verbose = 0)

        return action_val

    def update_epsilon(self, increase_epsilon: float = 0.0):
        if increase_epsilon > 0.0:
            self.epsilon+=increase_epsilon
        elif self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
