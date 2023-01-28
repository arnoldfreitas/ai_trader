import numpy as np
import pandas as pd
import random
import math
import os
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List

class BTCMarket_Env():
    '''
    Environment Class.
    
    Interaction of the agent with the environment. Calculates utility/reward. Observes states and executes actions (env.step()).
    '''
    def __init__(self,
                observation_space: tuple, # (5, 15)
                action_space: tuple,
                start_money: float,
                trading_fee: float = 0,
                asset: str = 'BTC',
                onefile: bool = False,
                ep_period: int = 2*24*14,
                ep_data_cols: List[str]=['close','histogram','50ema','rsi14'],
                RL_Algo: str = 'DQN',
                data_path: str ='./../data',) -> None:
        """
        Receive arguments and initialise the  class params.
        """
        # General Information Params
        self.ep_count = 0
        self.data_path = data_path
        self.data_source = self.load_data(onefile, asset)

        # Wallet Information Params:
        self.start_money = start_money
        self.inventory = None # postions: List[Tuple[Money_Invested, Unit_Price, Units]]
        self.money_available = None 
        self.wallet_value = None # money_available + BTC_price * units_in_inventory
        self.money_fiktiv = None

        # States / Observation
        self.observation_space = observation_space 
        self.state_size = observation_space[0] # amount of parameters
        self.window_size = observation_space[1] # amount of tome-steps from states
        
        # Action
        self.action_space = action_space

        # Episode Parameters
        self.ep_period = ep_period
        self.trading_fee = trading_fee
        self.ep_data_cols = ep_data_cols
        self.episode_length = 0
        self.ep_data = None

    def reset(self) -> None:
        """
        Restart/Start episodes 
        Parameters
        ----------
        period : int
        
        Notes
        -----
        It is embedded inside rl_agent.AI_Trader.train, when a episode is over and another starts
        """
        # Init internal Episode Params
        self.ep_timestep = 0
        # Episode Data
        self.ep_data = self.get_random_sample(self.data_source, period=self.ep_period)
        self.episode_length = len(self.ep_data)-1 
        self.ep_count+=1

        # Internal Wallet Information Params:
        self.windowed_money = [self.start_money]*(self.window_size+1)
        self.inventory = [] # positions: List[Tuple[Money_Invested, Average_Price, Units_Total]]
        self.money_available = self.start_money
        self.btc_wallet = 0 # Amount of BTCs in wallet
        self.wallet_value = self.start_money # money_available + BTC_price * units_in_inventory
        self.money_fiktiv = np.array([self.wallet_value])

        self.buy_count = 0
        self.sell_count = 0
        # the following are just for now until we decide 
        self.buy_long_count = 0
        self.sell_long_count = 0
        self.buy_short_count = 0
        self.sell_short_count = 0

    def step(self, 
            action: np.ndarray, 
            btc_wallet_variaton: float, # in [-1,1] 
            ) -> Tuple[np.ndarray, float, bool]:
        """
        Receives an action (Output from Agent.compute_action) and computes the next observation/state and its reward.
        
        Notes
        -----
        build state as in rl_agent.AI_Trader.state_creator
        """
        assert (self.ep_data is not None)

        actual_price = self.ep_data['close'].values[self.ep_timestep]

        # Compute Wallet States
        money_variaton = 0
        btc_invest = 0
        btc_units = 0
        if  btc_wallet_variaton > 0:
            self.buy_count += 1
            btc_invest = self.money_available * btc_wallet_variaton
            money_variaton = - btc_invest
            # fee = btc_invest * self.trading_fee
            # btc_invest-=fee
            btc_invest *= (1-self.trading_fee) 
            btc_units = btc_invest / actual_price
            self.inventory.append((btc_invest, 
                    actual_price,
                    btc_units))
        elif  btc_wallet_variaton < 0:
            self.sell_count += 1
            # TODO: Change to fit continuos Act_space. At the moment: sell all BTC in wallet:
            btc_wallet_variaton = -1
            money_variaton = sum([x[2] for x in self.inventory]) \
                * abs(btc_wallet_variaton) * actual_price ####### why sum()??
            btc_units = - money_variaton / actual_price
            money_variaton *= (1-self.trading_fee)
         
        # Compute State s(t+1)
        starting_id = self.ep_timestep - self.window_size
        if starting_id >= 0:
            windowed_close_data = self.ep_data['close'].values[starting_id:self.ep_timestep+1] 
            windowed_hist_data = self.ep_data['histogram'].values[starting_id:self.ep_timestep+1]
            windowed_ema_data = self.ep_data['50ema'].values[starting_id:self.ep_timestep+1]
            windowed_rsi_data = self.ep_data['rsi14'].values[starting_id:self.ep_timestep+1]
        else:
            windowed_close_data = [self.ep_data['close'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['close'].values[0:self.ep_timestep+1])
            windowed_hist_data = [self.ep_data['histogram'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['histogram'].values[0:self.ep_timestep+1])
            windowed_ema_data = [self.ep_data['50ema'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['50ema'].values[0:self.ep_timestep+1])
            windowed_rsi_data = [self.ep_data['rsi14'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['rsi14'].values[0:self.ep_timestep+1])

        state = []
        for i in range(self.window_size):
            state.append(self.sigmoid(windowed_close_data[i+1] - windowed_close_data[i]))
            state.append(self.sigmoid(windowed_hist_data[i]))
            state.append(self.sigmoid(windowed_ema_data[i+1] - windowed_ema_data[i]))
            state.append(self.sigmoid(self.windowed_money[i+1] - self.windowed_money[i]))
            state.append(windowed_rsi_data[i]/100)
        #state.append(money)
        state = np.array([np.nan_to_num(state)])
        # Compute Reward
        # reward = self.compute_reward(state, action, actual_price)
        reward = self.compute_utility(state, action, actual_price)
        # Change on inventory after reward, in case we sell
        if btc_wallet_variaton == -1:
            # TODO: Change to fit continuos Act_space. At the moment: sell all BTC in wallet:
            # TODO: We need to change inventory to fit selling only part of BTCs in wallet
            # Should the BTC-Wallet be a FIFO?
            self.inventory = []

        # At the end of step: necessary updates to internal params
        self.money_available += money_variaton
        self.btc_wallet += btc_units
        self.wallet_value = self.money_available + btc_invest
        # Check if Episode is Done
        if self.ep_timestep == self.episode_length - 1:
            done = True
        else:
            done = False
            # self.memory.append((state, action, reward, next_state, done))
            self.windowed_money.pop(0)
            # windowed_money.append(self.money_available)
            self.windowed_money.append(self.wallet_value)
            self.ep_timestep+=1

        return state, reward, done

    def step_continous_btc(self, 
            action: np.ndarray, 
            btc_wallet_variaton: float, # in [-1,1] 
            ) -> Tuple[np.ndarray, float, bool]:
        """
        Receives an action (Output from Agent.compute_action) and computes the next observation/state and its reward.
        
        Notes
        -----
        build state as in rl_agent.AI_Trader.state_creator
        """
        assert (self.ep_data is not None)
        
        actual_price = self.ep_data['close'].values[self.ep_timestep]

        # Compute Wallet States
        money_variaton = 0
        btc_invest = 0
        btc_partly_invest = 0
        btc_units = 0
        self.money_fiktiv = np.append(self.money_fiktiv, self.wallet_value)
        '''
        1: maximal buy in
        0: no position
        '''
        last_variation = (self.wallet - self.money_available) / self.wallet_value
        if btc_wallet_variaton > last_variation:
            # buy & buy more 
            self.buy_count += 1
            if last_variation == 0:
                btc_invest = self.money_available * btc_wallet_variaton
                money_variaton = - btc_invest
                btc_invest *= (1-self.trading_fee)
                average_price =  actual_price
            else:
                btc_partly_invest = self.wallet_value * btc_wallet_variaton - (self.wallet_value - self.money_available) # x anteile dazukaufen
                money_variaton = - btc_partly_invest
                btc_invest = self.inventory[-1][0] + btc_partly_invest*(1-self.trading_fee)
                average_price = (last_variation * self.inventory[-1][1] + btc_partly_invest * actual_price) / (last_variation + btc_partly_invest)
            btc_units = btc_invest / average_price
            self.inventory.append((btc_invest, 
                    average_price,
                    btc_units))

        elif btc_wallet_variaton < last_variation:
            # sell 
            self.sell_count += 1
            btc_partly_invest = self.wallet_value * btc_wallet_variaton - (self.wallet_value - self.money_available) # x anteile verkaufen
            btc_invest = self.inventory[-1][0] + btc_partly_invest
            money_variaton = - btc_partly_invest*(1-self.trading_fee)
            average_price = (last_variation * self.inventory[-1][1] + btc_partly_invest * actual_price) / (last_variation + btc_partly_invest)
            btc_units = btc_invest / average_price
            if btc_wallet_variaton != 0:
                self.inventory.append((btc_invest, 
                        average_price,
                        btc_units))
        
        # wie wird evaluiert ob hold/wait signale gut waren??? in inventory sthet nichts darüber drin
        
        # idee: self.inventory max.length = 2. Timestep t und t+1
        else:
            #TODO: hold, what do we do with inventory?: No change, leave as it as
            self.inventory.append((btc_invest, 
                        average_price,
                        btc_units))
        



        # else btc_wallet_variaton == last_variation --> hold/wait

        # Compute State s(t+1)
        starting_id = self.ep_timestep - self.window_size
        if starting_id >= 0:
            windowed_close_data = self.ep_data['close'].values[starting_id:self.ep_timestep+1]
            windowed_hist_data = self.ep_data['histogram'].values[starting_id:self.ep_timestep+1]
            windowed_ema_data = self.ep_data['50ema'].values[starting_id:self.ep_timestep+1]
            windowed_rsi_data = self.ep_data['rsi14'].values[starting_id:self.ep_timestep+1]
        else:
            windowed_close_data = [self.ep_data['close'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['close'].values[0:self.ep_timestep+1])
            windowed_hist_data = [self.ep_data['histogram'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['histogram'].values[0:self.ep_timestep+1])
            windowed_ema_data = [self.ep_data['50ema'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['50ema'].values[0:self.ep_timestep+1])
            windowed_rsi_data = [self.ep_data['rsi14'].values[0]]*abs(starting_id) \
                    + list(self.ep_data['rsi14'].values[0:self.ep_timestep+1])

        state = []
        for i in range(self.window_size):
            state.append(self.sigmoid(windowed_close_data[i+1] - windowed_close_data[i]))
            state.append(self.sigmoid(windowed_hist_data[i]))
            state.append(self.sigmoid(windowed_ema_data[i+1] - windowed_ema_data[i]))
            state.append(self.sigmoid(self.windowed_money[i+1] - self.windowed_money[i]))
            state.append(windowed_rsi_data[i]/100)
        #state.append(money)
        state = np.array([np.nan_to_num(state)])
        # Compute Reward
        reward = self.compute_reward_sterling_ratio(state, action, actual_price)


        # Change on inventory after reward, in case we sell
        '''if btc_wallet_variaton == 0 & self.last_invest_variation != 0: ## means: only if we just sold, update
            # TODO: Change to fit continuos Act_space. At the moment: sell all BTC in wallet:
            # TODO: We need to change inventory to fit selling only part of BTCs in wallet
            # Should the BTC-Wallet be a FIFO?
            self.inventory = []'''

        # At the end of step: necessary updates to internal params
        self.money_available += money_variaton
        self.btc_wallet += btc_units
        self.wallet_value = self.money_available + btc_invest
        # Check if Episode is Done
        if self.ep_timestep == self.episode_length - 1:
            done = True
        else:
            done = False
            # self.memory.append((state, action, reward, next_state, done))
            self.windowed_money.pop(0)
            # windowed_money.append(self.money_available)
            self.windowed_money.append(self.wallet_value)
            self.ep_timestep+=1

        return state, reward, done

    def compute_reward(self, state: np.ndarray, action: np.ndarray,
                actual_price: float,) -> float:
        """
        Function to compute reward based on state and action.

        Notes
        -----
        build state as in rl_agent.AI_Trader.get_reward_money

        Parameters
        ----------
        state: np.ndarray, 
        action: np.ndarray,
        actual_price: float
            Acutal BTC Price
        Returns
        -------
        reward
            Reward Value
        """

        # just a short example but we need to implement array of max profit/loss etc for calculating sharpe & sterling ratio

        past_profit = self.start_money - self.wallet_value # do we actually need it here?

        if action != 0:
            invest, buy_in, _ = self.inventory[-1]
            immediate_profit = invest * (state[-5] + actual_price) / buy_in # immediate profit form t to t+1.
            reward = immediate_profit
            # if position is 0 ( not invested), and we hold/wait the reward is always 0. Need to chenge that!!!
            # maybe its automatically good if we take sterling ratio or similar?
        else:
            reward = - state[-5] / actual_price * self.money_available # profit der gemacht hätte werden können

        return reward

    def compute_reward_sterling_ratio(self, state: np.ndarray, action: np.ndarray,
                actual_price: float,) -> float:
        """
        Function to compute reward based on state and action.

        Notes
        -----
        build state as in rl_agent.AI_Trader.get_reward_money

        Parameters
        ----------
        state: np.ndarray, 
        action: np.ndarray,
        actual_price: float
            Acutal BTC Price
        Returns
        -------
        reward
            Reward Value
        """

        # timestep t+1
        if action != 0:
            invest, buy_in, _ = self.inventory[-1]
            invest_new = invest * (state[-5] + actual_price) / buy_in # immediate profit form t to t+1.
            wallet_new = invest_new + self.money_available
        else:
            # case wait: not invested for t AND t-1
            # TODO: hold
            wallet_new = self.wallet_value - state[-5] / actual_price * self.money_available # profit der gemacht hätte werden können
            # state[-5] gibt veränderung btc kurs von t zu t+1 an. -> prozentuale veränderung berechnen & mit money_available verrechnen
            # MINUS: weil chance verpasst. Wenn kurs gefallen ist, ist state[-5] negativ -> wallet wird größer -> höherer reward, da wait gute aktion war


        fikitv_new = np.append(self.money_fiktiv, wallet_new)
        cummulative_return = (wallet_new - self.start_money) / self.start_money
        relative_drawdown = np.maximum.accumulate(fikitv_new) - fikitv_new
        absolute_drawdown = relative_drawdown / fikitv_new
        max_drawadown = np.max(absolute_drawdown)

        reward = cummulative_return / (max_drawadown - 0.1)

        return reward

    def sigmoid(self,x):
        try:
            result = math.exp(-x)
        except OverflowError:
            result = math.inf
        return 1 /(1 + result)
 
    def get_random_sample(self, data, period):
        start=random.randint(0,len(data)-period)
        return data.iloc[start:start+period,:] 
 
    def load_data(self, onefile=False,asset=None):
        out=[]
        for file in os.listdir('./../data'):
            if asset is not None and asset not in file:
                continue
            if 'histData_dt1800.0s' in file:
                out.append(pd.read_csv('./../data/'+file))
        if onefile:
            out_df=pd.DataFrame(columns=out[0].columns)
            for item in out:
                out_df=pd.concat([out_df,item])
            return out_df
        else:
            return out

    @staticmethod 
    def stock_price_format(n):
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