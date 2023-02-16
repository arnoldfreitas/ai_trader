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
        self.expected_return = None
        self.variance_returns_squared = None

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
        self.log_folder=os.path.abspath(os.path.join(self.data_path, 
                                        time_str, RL_Algo))
        self.log_dict = None

    def _update_log_folder(self, new_log_folder):
        self.log_folder=os.path.abspath(new_log_folder)

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
        # wallet_value = money_available + BTC_price * self.long_wallet[0] + BTC_Price*self.short_wallet[0]
        self.wallet_value = self.start_money + self.long_wallet[0]*self.long_wallet[1] + self.short_wallet[0]*self.short_wallet[1] # [euros]
        # position on btc in percentage of wallet value
        self.long_position: float = 0 # [%]: (self.long_wallet*BTC.price) / self.wallet_value[-1]
        self.short_position: float = 0 # [%]: (self.short_wallet*BTC.price) / self.wallet_value[-1]
        # Return history paramerter for reward computation
        self.expected_return = 0
        self.variance_returns_squared = 0

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
        action = np.round(action, 2) # reduce action to 2 decimals to avoid micro transactions
        # Change short/long position to actual_price in compare to wallet_value
        if self.long_position != 0 and self.short_position == 0:
            self.wallet_value = np.round(self.money_available + self.long_wallet[0]*actual_price, 2)
            self.long_position = np.round(self.long_wallet[0]*actual_price / self.wallet_value, 2)
        elif self.short_position != 0 and self.long_position == 0:
            self.wallet_value = np.round(self.money_available + self.short_wallet[0]*actual_price, 2)
            self.short_position = np.round(self.short_wallet[0]*actual_price / self.wallet_value, 2)

        # Compute Wallet States
        new_long_wallet, new_short_wallet, money_variation, trading_fee = self._handle_position(
                        new_position=action[0], 
                        btc_price=actual_price)
         
        # Compute State s(t+1)
        state = self._get_new_state()
        
        # Compute Reward
        reward = self.compute_reward_from_tutor(state, action, actual_price)

        # At the end of step: necessary updates to internal params
        self.money_available += money_variation
        self.money_available = np.round(self.money_available,2)
        self.long_wallet = new_long_wallet
        self.short_wallet = new_short_wallet
        # True wallet_value, either long or short will be 0
        self.wallet_value = np.round(self.money_available + new_long_wallet[0] * actual_price + new_short_wallet[0] * actual_price, 2)
        # Both positions have to be upgraded in case we sold and bought in one step
        long_val = np.round(new_long_wallet[0] * actual_price , 2)
        short_val = np.round(new_short_wallet[0] * actual_price , 2)
        self.long_position = np.round(long_val / self.wallet_value, 2)
        self.short_position = np.round(short_val / self.wallet_value, 2)
        test_action = np.round(new_long_wallet[0] * actual_price / self.wallet_value \
            + new_short_wallet[0] * actual_price / self.wallet_value, 2) # long or short will be zero
        if not( abs(test_action - action[0]) < 1e-2 ):
                print(f"Value not Expected after action:\n \
        action: {action}; test_action: {test_action};\n \
        long pos/ val {self.long_position} , {long_val};\n \
        short pos/ val {self.short_position} , {short_val};\n \
        money_available: {self.money_available}; wallet_value: {self.wallet_value}")
        self.windowed_money.pop(0)
        # windowed_money.append(self.money_available)
        self.windowed_money.append(self.wallet_value)
        self.ep_timestep+=1
        
        # Check if Episode is Done
        if self.ep_timestep == self.episode_length - 1:
            done = True
        else:
            done = False
    
        # Log Episode Step to log_dict
        self.log_episode_step(action = action, state = state, 
                reward = reward, done = done, closing_price = actual_price, 
                fee_paid = trading_fee, 
                btc_units = new_long_wallet[0] + new_short_wallet[0], 
                btc_eur= new_long_wallet[0]*new_long_wallet[1] + new_short_wallet[0]*new_short_wallet[1])

        return state, reward, done

    def init_logging_dict(self) -> dict:
        self.log_cols=['episode', 'action', 'state', 'reward', 'done','money',
            'btc_units','btc_eur','fee_paid', 'btc_price',  'long_wallet', 'short_wallet', 
            'wallet_value', 'long_position', 'short_position', 'buy_long_count', 
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
            btc_wallet_variation:float, 
            btc_price:float,
            new_position:float, 
            tmp_long_position:float,
            tmp_wallet_value:float,
            ):
        """
        Notes
        -----
        Action Defines how much % of wallet invested in Crypto we need to have at the end of this step.
        The amount to be traded depends on the fee, because the fee reduce the value of wallet, but
        the fee by itself depends on the amount to be traded, thus the value of btc at the end of step is computed by:
            btc_in_wallet_at_end_of_step = (new_position*(wallet_value + np.sign(btc_wallet_variation) * fee_rate * value_of_btc_in_wallet)) / 
                                        (1+np.sign(btc_wallet_variation) * fee_rate*new_position) 
        Information about inner params:
            long_variation_eur & long_variation_units_BTC > 0
                & money_variation < 0: Buy
            long_variation_eur & long_variation_units_BTC < 0
                & money_variation > 0: Sell
            long_variation_eur & long_variation_units_BTC 
                & money_variation == 0: Hold

        Paramters:
        ----------
        btc_wallet_variation:float 
            Variation in wallet.
            btc_wallet_variation > 0: Buy
            btc_wallet_variation < 0: Sell
            btc_wallet_variation == 0: Hold
        btc_price:float
            Price of bitcoin at t
        """
        #  [euros] Amount of euros of BTC to at end of step
        long_at_end_of_step = (new_position * \
            (tmp_wallet_value + np.sign(btc_wallet_variation) * self.trading_fee * \
                    tmp_long_position * tmp_wallet_value)) / \
                    (1+np.sign(btc_wallet_variation) * self.trading_fee * new_position) 
        #  [euros] Amount of euros of BTC to trade in order to have the given final position
        long_variation_eur = long_at_end_of_step - tmp_long_position*tmp_wallet_value
        # [units of btc] Amount of BTC bought with long_invest_eur
        long_variation_units_BTC = long_variation_eur / btc_price
        # Fees paid
        trading_fee = abs(long_variation_eur*self.trading_fee)
        # [euros] Actual amount of euros that go out of or into wallet
        money_variation = - long_variation_eur - trading_fee
        
        new_amount_btc_in_wallet = self.long_wallet[0] + long_variation_units_BTC
        if new_amount_btc_in_wallet < 1e-3:
            new_avg_price_btc_in_wallet = 0.0 
        else:    
            new_avg_price_btc_in_wallet = (self.long_wallet[1]*self.long_wallet[0] + long_variation_eur) / new_amount_btc_in_wallet
        # self.long_wallet = [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet]
        if (btc_wallet_variation > 1e-3): # BUY
            self.buy_long_count += 1
            if not(money_variation <  0 ) or not(long_variation_eur > 0):
                print(f"Value not Expected for buy position:\n \
        btc_wallet_variation ; {btc_wallet_variation}\n \
        money_variation: <0 ; {money_variation}\n \
        long_variation_eur: >0 ; {long_variation_eur}\n \
        new_amount_btc_in_wallet: {self.long_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.long_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet], \
                        money_variation, trading_fee
        elif (btc_wallet_variation < -1e-3): # SELL
            self.sell_long_count += 1
            if not(money_variation >  0 ) or not(long_variation_eur < 0):
                print(f"Value not Expected for sell position:\n \
        btc_wallet_variation ; {btc_wallet_variation}\n \
        money_variation: >0 ; {money_variation}\n \
        long_variation_eur: <0 ; {long_variation_eur}\n \
        new_amount_btc_in_wallet: {self.long_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.long_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet], \
                        money_variation, trading_fee
        else: # HOLD
            if not(abs(money_variation) <1e-3 ) or not(abs(long_variation_eur) < 1e-3):
                print(f"Value not Expected for holding position:\n \
        btc_wallet_variation ; {btc_wallet_variation}\n \
        money_variation: 0 ; {money_variation}\n \
        long_variation_eur: 0 ; {long_variation_eur}\n \
        new_amount_btc_in_wallet: {self.long_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.long_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return self.long_wallet, \
                        money_variation, trading_fee

    def _handle_short_position(self,
            btc_wallet_variation:float, 
            btc_price:float,
            new_position:float,
            tmp_short_position:float,
            tmp_wallet_value:float,
            ):
        """
        Notes
        -----
        Information about inner params:
            short_variation_eur & short_variation_units_BTC > 0
                & money_variation < 0: Buy
            short_variation_eur & short_variation_units_BTC < 0
                & money_variation > 0: Sell
            short_variation_eur & short_variation_units_BTC 
                & money_variation == 0: Hold

        Paramters:
        ----------
        btc_wallet_variation:float 
            Variation in wallet.
            btc_wallet_variation > 0: Buy
            btc_wallet_variation < 0: Sell
            btc_wallet_variation == 0: Hold
        btc_price:float
            Price of bitcoin at t
        """
        #  [euros] Amount of euros of BTC to at end of step
        short_at_end_of_step = (new_position * \
            (tmp_wallet_value + np.sign(btc_wallet_variation) * self.trading_fee * \
                tmp_short_position * tmp_wallet_value)) / \
                                        (1+np.sign(btc_wallet_variation) * self.trading_fee * new_position) 
        #  [euros] Amount of euros of BTC to trade in order to have the given final position
        short_variation_eur = short_at_end_of_step - tmp_short_position*tmp_wallet_value
        # [units of btc] Amount of BTC bought with short_invest_eur
        short_variation_units_BTC = short_variation_eur / btc_price
        # Fees paid
        trading_fee = abs(short_variation_eur*self.trading_fee)
        # [euros] Actual amount of euros that go out of or into wallet
        money_variation = - btc_wallet_variation - trading_fee

        new_amount_btc_in_wallet = self.short_wallet[0] + short_variation_units_BTC
        if new_amount_btc_in_wallet == 0:
            new_avg_price_btc_in_wallet = 0.0 
        else:    
            new_avg_price_btc_in_wallet = (self.short_wallet[1]*self.short_wallet[0] + short_variation_eur) / new_amount_btc_in_wallet

        if (btc_wallet_variation > 0): # BUY
            self.buy_short_count += 1
            if not(money_variation <  0 ) or not(short_variation_eur > 0):
                print(f"Value not Expected for holding position:\n \
        money_variation: <0 ; {money_variation}\n \
        short_variation_eur: >0 ; {short_variation_eur}\n \
        new_amount_btc_in_wallet: {self.short_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.short_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet], \
                        money_variation, trading_fee
        elif (btc_wallet_variation < 0): # SELL
            self.buy_short_count += 1
            if not(money_variation >  0 ) or not(short_variation_eur < 0):
                print(f"Value not Expected for holding position:\n \
        money_variation: >0 ; {money_variation}\n \
        short_variation_eur: <0 ; {short_variation_eur}\n \
        new_amount_btc_in_wallet: {self.short_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.short_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return [new_amount_btc_in_wallet, new_avg_price_btc_in_wallet], \
                        money_variation, trading_fee
        else: # HOLD
            if not(money_variation == 0 )or not(short_variation_eur == 0):
                print(f"Value not Expected for holding position:\n \
        money_variation: 0 ; {money_variation}\n \
        short_variation_eur: 0 ; {short_variation_eur}\n \
        new_amount_btc_in_wallet: {self.short_wallet[0]} ; {new_amount_btc_in_wallet}\n \
        new_avg_price_btc_in_wallet: {self.short_wallet[1]} ; {new_avg_price_btc_in_wallet}\n")
            return self.short_wallet, \
                        money_variation, trading_fee
    
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
        #TODO: HOW DO WE LOG IN CASE WE HAVE TO SELL FIRST AND THEN BUY IN ONE STEP. I think should be no problem
        #TODO: 2 different prices for long and short? have to look this up and change in code if needed
        # Still not handling the case we have no money to pay fees after trade. ? impossible scenario, fees are payed from trade money
        #TODO: Are the returns alright inside if and elif? yes if we make sure that for all situations there is a return
        #TODO: check case wait

        new_position = np.clip(new_position, -1, 1)

        # Init temporary vars
        tmp_wallet_value = self.wallet_value
        tmp_short_position = self.short_position
        tmp_long_position = self.long_position
        trade_tol = 0.05

        # Init output vars
        trading_fee_out = 0 
        new_short_wallet_out = [0,0] 
        new_long_wallet_out = [0,0] 
        money_variation_out = 0

        ### How does this work?? if new_position = 0.5 and old position was -0.5 it wont work i guess
        # sell all positions if from short to long, from long to short or new_position = 0
        if  abs(new_position) <= 1e-3:
            if self.short_position > 1e-3:
                # sell all shorts first. later in this function by longs with the amount of new_position
                btc_wallet_variation = - self.short_position
                new_short_wallet, money_variation, trading_fee = self._handle_short_position(
                        btc_wallet_variation=btc_wallet_variation, 
                        btc_price=btc_price, 
                        new_position=new_position,
                        tmp_short_position = tmp_short_position,
                        tmp_wallet_value = tmp_wallet_value)
                
                trading_fee_out += trading_fee
                new_short_wallet_out = new_short_wallet
                money_variation_out += money_variation

            if self.long_position > 1e-3:
                # sell all and keep going with rest of the function
                btc_wallet_variation = - self.long_position
                new_long_wallet, money_variation, trading_fee = self._handle_long_position(
                        btc_wallet_variation=btc_wallet_variation, 
                        btc_price=btc_price, 
                        new_position=new_position,
                        tmp_long_position = tmp_long_position,
                        tmp_wallet_value = tmp_wallet_value)

                trading_fee_out += trading_fee
                new_long_wallet_out = new_long_wallet
                money_variation_out += money_variation
                
            #case: WAIT, we havent been invested and want to stay NOT INVESTED
            return new_long_wallet_out, new_short_wallet_out, money_variation_out, trading_fee_out


        if new_position > 1e-3: # Adopting a LONG Position
            # only compute new btc_variation if we are already long. If we were short before, we coputed this value already above
            # self.long_position is the last position we adopted. Imagine the position last timestep was 0.4. If the btc_price rised, 
            # the position in compare to our wallet value got "bigger" by percent (e.g. 0.44).
            # We allow our network to give actions with 1 decimal point only. When do we hold??
            if abs(new_position - (self.long_wallet[0]*btc_price) / self.wallet_value) < trade_tol:
                # in case the change in investment is smaller than 0.1% of wallet_value, we hold
                # return self.long_wallet, new_short_wallet_out, money_variation_out, trading_fee_out
                return self.long_wallet, self.short_wallet, 0, 0
            
            # in case we had to sell shorts before buying longs, ad trading_fee from before
            if abs(self.short_position) > 1e-3:
                # sell all shorts first. later in this function buy longs with the amount of new_position
                btc_wallet_variation = - self.short_position
                new_short_wallet, money_variation, trading_fee = self._handle_short_position(
                        btc_wallet_variation=btc_wallet_variation, 
                        btc_price=btc_price, 
                        new_position=new_position,
                        tmp_short_position = tmp_short_position,
                        tmp_wallet_value = tmp_wallet_value)
                
                trading_fee_out += trading_fee
                new_short_wallet_out = new_short_wallet
                money_variation_out += money_variation

                tmp_wallet_value = np.round(self.money_available + money_variation_out + self.long_wallet[0]*self.long_wallet[1] + new_short_wallet_out[0]*new_short_wallet_out[1], 2)
                tmp_long_position = np.round((self.long_wallet[0]*self.long_wallet[1])/tmp_wallet_value, 2)
            
            btc_wallet_variation = new_position - round((self.long_wallet[0]*btc_price) / self.wallet_value, 2)

            if abs(btc_wallet_variation) > 1:
                print(f"Handle position for long. btc_wallet_variation > 1: {btc_wallet_variation}")
            new_long_wallet, money_variation, trading_fee = self._handle_long_position(
                        btc_wallet_variation=btc_wallet_variation, 
                        btc_price=btc_price, 
                        new_position=new_position,
                        tmp_long_position = tmp_long_position,
                        tmp_wallet_value = tmp_wallet_value)
            
            trading_fee_out += trading_fee
            new_long_wallet_out = new_long_wallet
            money_variation_out += money_variation
            if np.round(self.money_available + money_variation_out, 2) < 0:
                print(f"Transaction denied due to lack of available money: {self.money_available} < {np.round(abs(money_variation_out), 2)}")
                return self.long_wallet, self.short_wallet, 0, 0
            return new_long_wallet, new_short_wallet_out, money_variation, trading_fee

        if new_position < 1e-3: # Adopting a SHORT Position
            # compute btc_wallet_variation, if we are already short and want to buy more short or sell some shorts 
            if abs(new_position - (self.short_wallet[0]*btc_price) / self.wallet_value) < trade_tol:
                # in case the change in investment is smaller than 0.1% of wallet_value, we hold
                return self.long_wallet, self.short_wallet, 0, 0
            
            if abs(self.long_position) > 1e-3:
                # sell all and keep going with rest of the function
                btc_wallet_variation = - self.long_position
                new_long_wallet, money_variation, trading_fee = self._handle_long_position(
                        btc_wallet_variation=btc_wallet_variation, 
                        btc_price=btc_price, 
                        new_position=new_position,
                        tmp_long_position=tmp_long_position,
                        tmp_wallet_value=tmp_wallet_value)

                trading_fee_out += trading_fee
                new_long_wallet_out = new_long_wallet
                money_variation_out += money_variation

                tmp_wallet_value = np.round(self.money_available + money_variation_out + self.short_wallet[0]*self.short_wallet[1] + new_long_wallet[0]*new_long_wallet[1], 2)
                tmp_short_position = np.round((self.short_wallet[0]*self.short_wallet[1])/tmp_wallet_value, 2)

            ### we negate the position for shorts, so that positive value means we buy shorts and negative means we sell shorts
            new_position = abs(new_position)
    
            btc_wallet_variation = new_position - round((self.short_wallet[0]*btc_price) / self.wallet_value, 2)
            if abs(btc_wallet_variation) > 1:
                print(f"btc_wallet_variation > 1: {btc_wallet_variation}")
            new_short_wallet, money_variation, trading_fee = self._handle_short_position(
                        btc_wallet_variation=btc_wallet_variation, 
                        btc_price=btc_price, 
                        new_position=new_position,
                        tmp_wallet_value = tmp_wallet_value,
                        tmp_short_position = tmp_short_position)

            trading_fee_out += trading_fee
            new_short_wallet_out = new_short_wallet
            money_variation_out += money_variation

            if np.round(self.money_available + money_variation_out, 2) < 0:
                print(f"Transaction denied due to lack of available money: {self.money_available} < {np.round(abs(money_variation_out), 2)}")
                return self.long_wallet, self.short_wallet, 0, 0

            return new_long_wallet, new_short_wallet_out, money_variation, trading_fee

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
        return np.array([np.nan_to_num(state)])

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
                actual_price: float) -> float:
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

        pos_yield = 0.0
        if self.long_wallet[1] > 0.0:
            pos_yield = actual_price / self.long_wallet[1]
        else:
            pos_yield = 1 - self.wallet_value / self.start_money
        return pos_yield

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
        #TODO: How do we handle the reward function with our action?

        wallet_history = self.log_dict['wallet_value']

        # wallet_value for timestep t+1
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

        fikitv_new = np.append(wallet_history, wallet_new)
        cummulative_return = (wallet_new - self.start_money) / self.start_money
        relative_drawdown = np.maximum.accumulate(fikitv_new) - fikitv_new
        absolute_drawdown = relative_drawdown / fikitv_new
        max_drawadown = np.max(absolute_drawdown)

        reward = cummulative_return / (max_drawadown - 0.1)

        return reward

    def reward_DSR(self, state: np.ndarray, action: np.ndarray,
                actual_price: float,) -> float:
        """
        Function to compute Differential Sharpe Ratio.

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
        # Look at Paper DRRL-Agent and Link in Word in googledrive for calculation dateils
        # TODO: tau_decay, exponential decay parameter. IMPLEMENT IN TRAINER CLASS I GUESS.
        tau_decay = 0.004
        # execution cost should be the difefrence between bid and ask + trading_fee
        execution_cost = action * self.wallet_value * self.trading_fee # + spread (bid-ask)
        actions = self.log_dict['action']
        old_action = actions[-1]
        old_price = actual_price
        #TODO: new_price bzw. price_change. state[-5] is actually the price change wrapped by a sigmoid function.
        new_price = state[-5]
        # absolute price change
        price_change = new_price - old_price

        # net return in an abstract manner, these are not exactly the immediate_profits, but represent positive and negative rewards
        # makes sense because the trader has to learn to trade with percentages of his wallet. He should get more profit just from having 
        # a higher amout of star_money     
        reward = price_change * action - execution_cost * abs(action - old_action) # - funding_cost * action
        self.variance_returns_squared = tau_decay * self.variance_returns_squared + (1 - tau_decay) * (reward - self.expected_return)**2
        self.expected_return = tau_decay * self.expected_return + (1 - tau_decay) * reward
        
        ###### THIS IS THE TARGET, maximum profit achievable. Our model tries to optimize to make these profits
        baseline_profit = abs(price_change) # - fee 
        # risk appetite parameter
        # Explanation: Mit unserem wallet_value könnten wir nicht zwingend soviel geld gemacht haben wie price_change angbt. 
        # Bei betrachtung unserer reward function, macht diese implementation aber sinn. 
        gamma = 252**0.5 * (self.expected_return - baseline_profit) / self.variance_returns_squared
        utility = self.expected_return - gamma / 2 * self.variance_returns_squared

        return utility

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