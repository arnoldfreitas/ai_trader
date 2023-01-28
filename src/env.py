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
        self.inventory = [] # positions: List[Tuple[Money_Invested, Unit_Price, Units]]
        self.money_available = self.start_money
        self.btc_wallet = 0 # Amount of BTCs in wallet
        self.wallet_value = self.start_money # money_available + BTC_price * units_in_inventory

        self.buy_count = 0
        self.sell_count = 0
        # the following are just for now until we decide 
        self.buy_long_count = 0
        self.sell_long_count = 0
        self.buy_short_count = 0
        self.sell_short_count = 0
        self.last_invest_variation = 0

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
        '''
        1: maximal buy in
        0: no position
        '''
        #### Fragen: ist wallet_value auf dem timestep stand von self.inventory[-1] ???
        # TODO: eventuell bessere lösung als self.last_invest_variation = x ??
        if btc_wallet_variaton > self.last_invest_variation:
            # buy & buy more 
            self.buy_count += 1
            if self.last_invest_variation == 0:
                btc_invest = self.money_available * btc_wallet_variaton
                money_variaton = - btc_invest
                btc_invest *= (1-self.trading_fee) 
            else:
                #btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][0]
                btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][2] * actual_price # x anteile dazukaufen
                money_variaton = - btc_partly_invest
                btc_invest = self.inventory[-1][0] + btc_partly_invest*(1-self.trading_fee) 
            btc_units = btc_invest / actual_price
            self.inventory.append((btc_invest, 
                    actual_price,
                    btc_units))

        elif btc_wallet_variaton < self.last_invest_variation:
            # sell 
            self.sell_count += 1
            #btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][0]
            btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][2] * actual_price # x anteile verkaufen
            money_variaton = - btc_partly_invest
            btc_invest = self.inventory[-1][0] + money_variaton
            money_variaton *= (1-self.trading_fee) 
            btc_units = self.inventory[-1][0] + btc_partly_invest / actual_price
            if btc_wallet_variaton != 0:  ##### if we sell the whole position, update inventory at the end?
                self.inventory.append((btc_invest, 
                        actual_price,
                        btc_units))
        self.last_invest_variation = btc_wallet_variaton

        # else btc_wallet_variaton == self.last_invest_variation --> hold/wait

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
        reward = self.compute_reward(state, action, actual_price)
        # Change on inventory after reward, in case we sell
        if btc_wallet_variaton == 0 & self.last_invest_variation != 0: ## means: only if we just sold, update
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

    def step_continous_perpetual_swap(self, 
    ##### not finished, need more information about the perpetual swap (prices etc) ######
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
        '''
        1: maximal buy in
        0: no position
        '''
        if  btc_wallet_variaton > 0:
            # TODO: eventuell bessere lösung als self.last_wallet_variation = x ??
            if self.last_invest_variation == 0:
                # buy in long
                self.buy_long_count += 1
                btc_invest = self.money_available * btc_wallet_variaton
                money_variaton = - btc_invest
                btc_invest *= (1-self.trading_fee) 
                btc_units = btc_invest / actual_price
                self.inventory.append((btc_invest, 
                        actual_price,
                        btc_units))
                self.last_invest_variation = btc_wallet_variaton
            elif btc_wallet_variaton > self.last_invest_variation:
                # buy more long
                self.buy_long_count += 1
                btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][0] # x dazukaufen
                money_variaton = - btc_partly_invest
                btc_invest = self.inventory[-1][0] + btc_partly_invest*(1-self.trading_fee) 
                btc_units = btc_invest / actual_price
                self.inventory.append((btc_invest, 
                        actual_price,
                        btc_units))
                self.last_invest_variation = btc_wallet_variaton
            else:
                # sell partly long
                self.sell_long_count += 1
                btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][0]
                money_variaton = - btc_partly_invest
                btc_invest = self.inventory[-1][0] + money_variaton
                money_variaton *= (1-self.trading_fee) 
                btc_units = self.inventory[-1][0] + btc_partly_invest / actual_price
                self.inventory.append((btc_invest, 
                        actual_price,
                        btc_units))
                self.last_invest_variation = btc_wallet_variaton
            # if btc_wallet_variaton = self.last_invest_variation --> hold/wait

        '''elif btc_wallet_variaton < 0:
            if self.last_invest_variation == 0:
                # buy in short
                self.buy_short_count += 1
                btc_invest = self.money_available * btc_wallet_variaton
                money_variaton = - btc_invest
                btc_invest *= (1-self.trading_fee) 
                btc_units = btc_invest / actual_price
                self.inventory.append((btc_invest, 
                        actual_price,
                        btc_units))
                self.last_invest_variation = btc_wallet_variaton
            elif btc_wallet_variaton > self.last_invest_variation:
                # buy more short
                self.buy_short_count += 1
                btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][0] # x dazukaufen
                money_variaton = - btc_partly_invest
                btc_invest = self.inventory[-1][0] + btc_partly_invest*(1-self.trading_fee) 
                btc_units = btc_invest / actual_price
                self.inventory.append((btc_invest, 
                        actual_price,
                        btc_units))
                self.last_invest_variation = btc_wallet_variaton
            else:
                # sell partly short
                self.sell_short_count += 1
                btc_partly_invest = self.wallet_value * btc_wallet_variaton - self.inventory[-1][0]
                money_variaton = - btc_partly_invest
                btc_invest = self.inventory[-1][0] + money_variaton
                money_variaton *= (1-self.trading_fee) 
                btc_units = self.inventory[-1][0] + btc_partly_invest / actual_price
                self.inventory.append((btc_invest, 
                        actual_price,
                        btc_units))
                self.last_invest_variation = btc_wallet_variaton



            self.sell_count += 1
            # TODO: Change to fit continuos Act_space. At the moment: sell all BTC in wallet:
            btc_wallet_variaton = -1
            money_variaton = sum([x[2] for x in self.inventory]) \
                * abs(btc_wallet_variaton) * actual_price
            btc_units = - money_variaton / actual_price
            money_variaton *= (1-self.trading_fee)
        '''
         
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
        reward = self.compute_reward(state, action, actual_price)
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

    def compute_valid_action(self, action: float) -> float:
        """
        Receives action_value from step() with one decimal point rounded in Agent.comput_action.

        Notes
        -----
        Should calculate a rational action.
        Should take in to account: open_position, free_money etc. from current state.

        Ideas:
        If action_val > 0: Invest (action_val) % of total money. If already invested with (action_val) % of total money, and at next timestep we get 
        a different action_val, change the position to new action_val % of total money.
        If action_val = 0: wait/hold.
        If action_val < 0: sell open position (if there is one) and go short with (action_val) % of total money

        Problem: Many Transaction fees, but the Agent would learn to cope with this problem maybe?! 
        
        """

        pass

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

        ### what exactly does this reward function calculate? net profit?

        wallet_value_variation= 0
        past_pofit = self.start_money - self.money_available ### not self.wallet_value ? if we made profit, the variable past_profit is negative
        # past_pofit = self.wallet_value - self.start_money
        for invest, buy_in, _ in self.inventory:
            pos_yield=(actual_price-buy_in)/buy_in
            wallet_value_variation+=invest+(invest*pos_yield)
        
        ### only if action != 0:
        ### why transaction_fee here? we already took it into accout in step()

        # fee = wallet_value_variation*self.trading_fee
        # wallet_value_variation -= fee
        wallet_value_variation *= (1-self.trading_fee)
        wallet_value_variation += self.money_available

        reward = wallet_value_variation - self.start_money + past_pofit
        # reward = wallet_value_variation - self.start_money - past_pofit
        # the reward is the profit just made from timestep t to t+1 with the action generated at timestep t. We dont take previous profit into account here for now.
        return reward

    def compute_utility(self, daily_profits: pd.DataFrame) -> np.ndarray:
        """
        To compute e.g. the sterling ratio, keep track of profit and loss of every trade (or daily prfoit) (preferably DataFrame Object)).

        Notes
        -----
        
        """

        pass

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