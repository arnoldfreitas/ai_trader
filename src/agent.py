import random
import os
import traceback
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# tf.compat.v1.disable_eager_execution()
import gc
from typing import Tuple
from env import BTCMarket_Env

np.random.seed(42)
random.seed(42)


class Trader_Agent():
    '''
    Trader class.

    Build model/load model, calculates action.

    '''
    
    def __init__(self, 
                observation_space: Tuple[int, int], 
                action_space: float ,
                action_domain: Tuple[float, float] = (0.0, 1.0), # min max
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
        # action_space[0] = 1 = dimension, action_space[1] = [-1, 1] = range of output of policy network and for random exploring

        #TODO: action_space as input important? 
        self.output_dim = action_space[0] ### Überhaupt wichtig hier action_space zu übergeben? Ist eigentlich immer derselbe
        self.action_domain = action_space[1] ### Same here
        '''
        Alternativ:

        self.output_dim = 1
        self.action_domain = [-1,1]
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
        self.action_domain = action_domain
        
        # Exploration Params
        self.epsilon = epsilon # exploration or not 1 is full random, start with 1 
        self.epsilon_final = epsilon_final # final epsilon
        self.epsilon_decay = epsilon_decay 

        # Model Params. Build/Load Model
        # Moving this part for trainer so it can be controlled from outside
        # self.model_name = model_name
        # if (load_model is not None) and (isinstance(load_model, str)):
        #     try:
        #         self.load_model()
        #     except:
        #         print('Model loading failed:')
        #         print(traceback.print_exc())
        #         self.build_model()
        # else:
        #     self.build_model()

    def build_model(self, learning_rate=1e-3, 
                        loss_function='mse',
                        use_softmax=False,):
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
        
        ### Continuous action space with MLP: Define the Percentage of wallet on Bitcoins 
        if min(self.action_domain) < 0.0:
            # for trading perpetual swap: "tanh"; action_domain in (-1, 1)
            layer_output = 'tanh'
        else :
            # for BTC: "sigmoid"; action_domain in (0, 1)
            layer_output = 'sigmoid'

        if use_softmax:
            model = keras.models.Sequential([
                keras.layers.InputLayer(input_shape=(self.window_size,self.state_size)),
                keras.layers.Flatten(),
                keras.layers.Dense(units=256, activation='relu'),
                keras.layers.Dense(units=128, activation='relu'),
                keras.layers.Dense(units=64, activation='relu'),
                keras.layers.Dense(units=self.action_space, activation=layer_output),
                keras.layers.Softmax()
                ])
        else:
            model = keras.models.Sequential([
                keras.layers.InputLayer(input_shape=(self.window_size,self.state_size)),
                keras.layers.Flatten(),
                keras.layers.Dense(units=256, activation='relu'),
                keras.layers.Dense(units=128, activation='relu'),
                keras.layers.Dense(units=64, activation='relu'),
                keras.layers.Dense(units=self.action_space, activation=layer_output)
                ])
        #TODO: Build RNN (LSTM) as policy network
        model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        model.summary()
        print(f"Model Loss: {model.compiled_loss._losses}")
        
        self.model = model
    
    def build_model_LSTM(self, learning_rate=1e-3, 
                         loss_function='mse',
                         lstm_path=None,
                         use_softmax=False,):
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
        
        ### Continuous action space with MLP: Define the Percentage of wallet on Bitcoins 
        if min(self.action_domain) < 0.0:
            # for trading perpetual swap: "tanh"; action_domain in (-1, 1)
            layer_output = 'tanh'
        else :
            # for BTC: "sigmoid"; action_domain in (0, 1)
            layer_output = 'sigmoid'

        input_layer = keras.layers.Input(shape=(self.window_size,self.state_size))
        lstm_inputs = keras.layers.Lambda(lambda x: x[:,:,4:], 
        # lstm_inputs = keras.layers.Lambda(lambda x: tf.expand_dims(x[:,-1,4:], 1), 
                                name="lstm_inputs")(input_layer)
        lstm = keras.models.load_model(lstm_path, compile=False)
        lstm.trainable=False
        lstm_layer = lstm(lstm_inputs)
        flaten_inputs = keras.layers.Flatten()(input_layer)
        lstm_outputs = keras.layers.Flatten()(lstm_layer)
        dense_inputs = keras.layers.concatenate([flaten_inputs,
                            tf.reshape(lstm_outputs, 
                            shape=(-1, 10))])
                            # shape=(-1, 5))])
        dense_1 = keras.layers.Dense(units=256, activation='linear')(dense_inputs)
        dense_2 = keras.layers.Dense(units=128, activation='linear')(dense_1)
        dense_3 = keras.layers.Dense(units=64, activation='linear')(dense_2)
        output = keras.layers.Dense(units=self.action_space, activation=layer_output)(dense_3)
        if use_softmax:
            output = keras.layers.Softmax()(output)
        self.model = keras.Model(input_layer, output)
        #TODO: Build RNN (LSTM) as policy network
        self.model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        self.model.summary()
        # print(f"Model Loss: {self.model.compiled_loss._losses}")
        
        # self.model = model

        # self.lstm_model = keras.Model(input_layer, lstm_layer)
        # self.lstm_model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def load_model(self, model_path: str, 
                         learning_rate=1e-3, 
                         loss_function='mse',):
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
        
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        self.model.summary()

    def compute_action(self, state: np.ndarray) ->  float:
        """
        Performs Trade action from a given state. Uses epsilon greedy method or model.predict to define action.
        
        Note
        ----
        Old implementaion of this method: As in rl_agent.AI_Trader.trade

        Returns
        -------
            Computed action
        """
        if random.random() <= self.epsilon:
            action = []
            for _ in range(self.action_space):
                action.append(random.uniform(*self.action_domain))

            if self.action_space > 1: # DQN
                out = tf.nn.softmax(np.array(action))
                out= out.numpy()
            else:  # DRL
                out = np.array(action)

            gc.collect()
            return out
      
        # action_val = self.model.predict(tf.reshape(tf.convert_to_tensor(state[0],dtype=np.float32),shape=(1,self.state_size*self.window_size)),verbose = 0)
        state_input = tf.convert_to_tensor(state, dtype=tf.float32)
        # print(tf.shape(state_input))
        # action_val = self.model.predict(state_input,verbose = 0,steps=1)[0]
        action_val = self.model(state_input, training=False)
        action_val = action_val.numpy()[0]
        gc.collect()
        # keras.backend.clear_session()

        # round computed value to one decimal point
        # leaving decision about rounding for trainer
        return action_val

    def update_epsilon(self, increase_epsilon: float = 0.0):
        if increase_epsilon > 0.0:
            self.epsilon+=increase_epsilon
        elif self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_final

        self.epsilon = min([self.epsilon, 1.0])
