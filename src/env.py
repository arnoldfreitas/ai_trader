import numpy as np
import pandas as pd
import random
import math
import os
from datetime import datetime
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
                ep_period: int = 2*24*14,
                ep_data_cols: List[str]=['close','histogram','50ema','rsi14'],
                RL_Algo: str = 'DQN',
                data_path: str ='./../data/',) -> None:
        """
        Receive arguments and initialise the  class params.
        """
        # General Information Params
        self.ep_count = 0
        self.data_path = data_path
        self.data_source = self.load_data(asset)
        self.len_source = len(self.data_source)
        # print(f"{type(self.data_source), type(self.data_source[0])}")
        # print(f"self.len_source: {self.len_source}")

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

        # Params for logging
        time_str=datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_folder=os.path.abspath(os.path.join(self.data_path, time_str, RL_Algo))
        self.log_dict = None

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
        self.ep_timestep: int = 0
        # Episode Data
        self.ep_data = self.get_random_sample(self.data_source, period=self.ep_period)
        self.episode_length: int = len(self.ep_data)-1 
        self.ep_count+=1

        # Internal Wallet Information Params:
        self.windowed_money: list = [self.start_money]*(self.window_size+1)
        self.inventory: list = [] # positions: List[Tuple[Money_Invested, Average_Price, Units_Total]]
        # money in wallet for future trade
        self.money_available: float = self.start_money # [euros]
        # Absolute amount of BTCs-Long in wallet
        self.long_wallet: list = [0, 0] # [units of longs, mean_price_bought]
        # Absolute amount of BTCs-Short in wallet
        self.short_wallet: list = [0, 0] # [units of shorts, mean_price_sold] 
        # money_available + BTC_price * self.long_wallet + BTC_Price*self.short_wallet
        self.wallet_value: float = self.start_money # [euros]:
        # TODO: why money_fiktiv? 
        self.money_fiktiv: list = np.array([self.wallet_value]) # [euros]
        # position on btc in percentage of wallet value
        self.long_position: float = 0 # [%]: (self.long_wallet*BTC.price) / self.wallet_value[-1]
        self.short_position: float = 0 # [%]: (self.short_wallet*BTC.price) / self.wallet_value[-1]

        # the following are just for now until we decide 
        self.buy_long_count: float = 0
        self.sell_long_count: float = 0
        self.buy_short_count: float = 0
        self.sell_short_count: float = 0

        # Init Logging
        self.log_dict = self.init_logging_dict()

    def step(self, 
            action: np.ndarray, # value in [-1,1] representing position of bitcoin to have in wallet on t+1
            ) -> Tuple[np.ndarray, float, bool]:
        """
        Receives an action (Output from Agent.compute_action) and computes the next observation/state and its reward.
        
        Notes
        -----
        build state as in rl_agent.AI_Trader.state_creator
        
        
        Parameters
        ----------
        action: numpy.ndarray of shape (1)
            Represent position of short or long to have in wallet on t+1.
            If the Policy gives a different output, then this must be handled by Trainer before calling env.step.  
        
        Returns
        -------
        state

        reward
        
        done
        """
        assert (self.ep_data is not None)

        actual_price = self.ep_data['close'].values[self.ep_timestep]
        # Compute Wallet States
        new_long_wallet , money_variaton, btc_eur_invest = self._handle_position(
                        new_position=action[0], 
                        btc_price=actual_price)
         
        # Compute State s(t+1)
        state = self._get_new_state()
        
        # Compute Reward
        reward = self.compute_reward_from_tutor(state, action, actual_price)

        # At the end of step: necessary updates to internal params
        self.money_available += money_variaton
        self.long_wallet = new_long_wallet
        self.wallet_value = self.money_available + new_long_wallet[0]*new_long_wallet[1]
        test_action = new_long_wallet[0]*new_long_wallet[1]/ self.wallet_value 
        if not( abs(test_action - action[0]) < 1e-2 ):
                print(f"Value no Expected after action:\n \
        action: {action[0]} ; {test_action}; {self.long_position}")
        self.long_position = test_action
            
        self.windowed_money.pop(0)
        # windowed_money.append(self.money_available)
        self.windowed_money.append(self.wallet_value)
        self.ep_timestep+=1
        
        # Check if Episode is Done
        if self.ep_timestep == self.episode_length - 1:
        # if self.ep_timestep > 5:
            done = True
        else:
            done = False
    
        # Log Episode Step to log_dict
        self.log_episode_step(action = action, state = state, 
                reward = reward, done = done, closing_price = actual_price, 
                fee_paid = abs(btc_eur_invest*self.trading_fee), 
                btc_units = new_long_wallet[0], 
                btc_eur= new_long_wallet[0]*new_long_wallet[1])

        return state, reward, done

    def init_logging_dict(self) -> dict:
        self.log_cols=['episode', 'action', 'state', 'reward', 'done','money',
            'btc_units','btc_eur','fee_paid', 'btc_price',  'long_wallet', 'short_wallet', 'wallet_value', 'long_position', 'short_position', 'buy_long_count', 
            'sell_long_count', 'buy_short_count', 'sell_short_count']
        return dict.fromkeys(self.log_cols, [])

    def log_episode_step(self, action, state, reward, done, 
                    closing_price, fee_paid, btc_units, btc_eur):
        """
        Add params to log dict
        """
        self.log_dict['episode'].append(self.ep_count)
        # self.log_dict['run'].append(0)
        self.log_dict['action'].append(action)
        self.log_dict['state'].append(state)
        self.log_dict['reward'].append(reward)
        self.log_dict['done'].append(done)
        self.log_dict['money'].append(self.money_available)
        self.log_dict['btc_units'].append(btc_units)
        self.log_dict['btc_eur'].append(btc_eur)
        self.log_dict['long_wallet'].append(self.long_wallet)
        self.log_dict['short_wallet'].append(self.short_wallet)
        self.log_dict['wallet_value'].append(self.wallet_value)
        self.log_dict['long_position'].append(self.long_position)
        self.log_dict['fee_paid'].append(fee_paid)
        self.log_dict['btc_price'].append(closing_price)
        self.log_dict['short_position'].append(self.short_position)
        self.log_dict['buy_long_count'].append(self.buy_long_count)
        self.log_dict['sell_long_count'].append(self.sell_long_count)
        self.log_dict['buy_short_count'].append(self.buy_short_count)
        self.log_dict['sell_short_count'].append(self.sell_short_count)

    def log_episode_to_file(self,episode=0,run=0):
        """
        Save log dict to CSV
        """
        
        os.makedirs(self.log_folder, exist_ok=True)
        
        df = pd.DataFrame.from_dict(self.log_dict)
        df.to_csv(self.log_folder + f"/Epi_{episode}_run_{run}.csv")

    def _handle_long_position(self,
            btc_wallet_variaton:float, 
            btc_price:float):
        """
        Notes
        -----
        Information about inner params:
            long_variation_eur & long_variation_units_BTC > 0
                & money_variation < 0: Buy
            long_variation_eur & long_variation_units_BTC < 0
                & money_variation > 0: Sell
            long_variation_eur & long_variation_units_BTC 
                & money_variation == 0: Hold

        Paramters:
        ----------
        btc_wallet_variaton:float 
            Variation in wallet.
            btc_wallet_variaton < 0: Buy
            btc_wallet_variaton > 0: Sell
            btc_wallet_variaton == 0: Hold
        btc_price:float
            Price of bitcoin at t
        """
        #TODO: Double check those calculation represent reality
        # [euros] Amount of euros of BTC to invest in order to have the given final position
        long_variation_eur = (btc_wallet_variaton * self.wallet_value)*(1-self.trading_fee)
        # [units of btc] Amount of BTC bought with long_invest_eur
        long_variation_units_BTC = long_variation_eur / btc_price
        # [euros] Actual amount of euros that go out of wallet
        fee_paid = abs(long_variation_eur*self.trading_fee)
        money_variation = -long_variation_eur-fee_paid
        new_amount_btc_in_wallet = self.long_wallet[0] + long_variation_units_BTC
        if new_amount_btc_in_wallet == 0:
            new_avg_price_btc_in_wallet = 0.0
        else:    
            new_avg_price_btc_in_wallet = (self.long_wallet[1]*self.long_wallet[0] + long_variation_eur) / new_amount_btc_in_wallet
        # self.long_wallet = [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet]
        if (btc_wallet_variaton > 0): # BUY
            self.buy_long_count += 1
            if not(money_variation <  0 ) or not(long_variation_eur > 0):
                print(f"Value no Expected for holding position:\n \
        money_variation: <0 ; {money_variation}\n \
        long_variation_eur: >0 ; {long_variation_eur}\n \
        new_amount_btc_in_wallet: {self.long_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.long_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet], \
                        money_variation, long_variation_eur
        elif (btc_wallet_variaton < 0): # SELL
            self.sell_long_count += 1
            if not(money_variation >  0 ) or not(long_variation_eur < 0):
                print(f"Value no Expected for holding position:\n \
        money_variation: >0 ; {money_variation}\n \
        long_variation_eur: <0 ; {long_variation_eur}\n \
        new_amount_btc_in_wallet: {self.long_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.long_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet], \
                        money_variation, long_variation_eur
        else: # HOLD
            if not(money_variation == 0 )or not(long_variation_eur == 0):
                print(f"Value no Expected for holding position:\n \
        money_variation: 0 ; {money_variation}\n \
        long_variation_eur: 0 ; {long_variation_eur}\n \
        new_amount_btc_in_wallet: {self.long_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.long_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return self.long_wallet, \
                        money_variation, long_variation_eur
    
    def _handle_position(self, 
            new_position:float, 
            btc_price:float)->None:
        """
        Parameters:
        ----------
        new_position:float 
            action[0]
        btc_price:float
            Price of bitcoin at t
        """
        #TODO: Step implemented to only handle longs

        if new_position > 0: # Adopting a LONG Position
            new_position = np.clip(new_position, 0,1)
            btc_wallet_variaton = new_position - self.long_position
            if abs(btc_wallet_variaton) > 1:
                print(f"btc_wallet_variaton > 1: {btc_wallet_variaton}")
            new_long_wallet , money_variaton, long_variation_eur = self._handle_long_position(btc_wallet_variaton=btc_wallet_variaton, 
                        btc_price=btc_price)
            return new_long_wallet, money_variaton, long_variation_eur

        elif new_position < 0: # Adopting a SHORT Position
            # TODO: implement SHORT
            return [0,0], 0, 0
        else: # OUT of ALL POSITIONS
            btc_wallet_variaton = - self.long_position
            new_long_wallet, money_variaton,long_variation_eur = self._handle_long_position(btc_wallet_variaton=btc_wallet_variaton, 
                        btc_price=btc_price)
            # TODO: implement SHORT
            return new_long_wallet, money_variaton, long_variation_eur

    def _get_new_state(self):
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
        return np.nan_to_num(state)

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
        # TODO: money_fiktive is like wallet_value, why the double variable? 
        # TODO: why always appending? 
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
        state = self._get_new_state()
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
   
    def compute_reward_from_tutor(self, state: np.ndarray, action: np.ndarray,
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


        pos_yield = actual_price / self.long_wallet[1]
        act_val = pos_yield * self.long_position
        
        fee=act_val*self.trading_fee
        act_val-=fee
        act_val+=self.money_available
        reward=act_val-self.start_money
        
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
 
    def get_random_sample(self, data_in, period):
        start=random.randint(0,len(data_in)-period)
        return data_in.iloc[start:start+period,:] 
 
    def load_data(self,asset=None,
            onefile=False, # kept for retro compatibility 
            ):
        out=[]
        for file in os.listdir('./../data'):
            if asset is not None and asset not in file:
                continue
            if 'histData_dt1800.0s' in file:
                out.append(pd.read_csv('./../data/'+file))
        
        out_df=pd.DataFrame(columns=out[0].columns)
        for item in out:
            out_df=pd.concat([out_df,item])
        return out_df

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