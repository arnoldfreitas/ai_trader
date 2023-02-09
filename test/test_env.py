import os
import sys
from pathlib import Path

import unittest
import numpy as np

BASE_FOLDER = Path(__file__).parent.parent.absolute().as_posix()
sys.path.append(os.path.join(BASE_FOLDER, "src"))
from env import BTCMarket_Env

class TestEnv(unittest.TestCase):
    
    def test_handle_long_position_increasing_position_by_10percent(self):
        # Start Env for Test
        btc_env = BTCMarket_Env(
                observation_space = (5, 20), # (5, 15)
                action_space = 1,
                start_money = 10000,
                trading_fee = 0.001)

        # Init values for testing, based on .reset
        btc_env.windowed_money = [btc_env.start_money]*(btc_env.window_size+1)
        btc_env.inventory = []
        btc_env.money_available = btc_env.start_money # [euros]
        btc_env.long_wallet = [0, 0] # [units of longs, mean_price_bought]
        btc_env.short_wallet = [0, 0] # [units of shorts, mean_price_sold] 
        btc_env.wallet_value = btc_env.start_money + btc_env.long_wallet[0]*btc_env.long_wallet[1] # [euros]
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [0.000, 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 
                0.600, 0.700, 0.800, 0.900, 1.000] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            btc_wallet_variation = action - btc_env.long_position

            new_long_wallet , money_variation, \
                fee_paid =btc_env._handle_long_position(
                    btc_wallet_variation=btc_wallet_variation,
                    btc_price = fixed_priced_for_test,
                    new_position = action,
                    tmp_wallet_value = btc_env.wallet_value,
                    tmp_long_position = btc_env.long_position)
            long_variation_eur = - money_variation - fee_paid
            # Update Vars as in step
            btc_env.money_available += money_variation
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variation)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([ 0.00, 0.00, -1000.90, -1000.70, -1000.50, 
                        -1000.30, -1000.10, -999.90, -999.70, -999.50, -999.30, -999.10])

        long_var_expected = np.array([0.00, 0.00, 999.90, 999.70, 999.50, 
                                999.30, 999.10, 998.90, 998.70, 998.50, 998.30, 998.10])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([[0.0, 0], [0.0, 0]],[[units, 100] for units in [10.00, 20.00, 29.99, 
                        39.98, 49.98, 59.96, 69.95, 79.94, 89.92, 99.90]]).flatten()

        self.assertTrue(np.allclose(pos_long_expected, long_pos_env, atol = 1e-2),
                            f'\nExpected:\n {pos_long_expected} \nComputed:\n {long_pos_env}')
        self.assertTrue(np.allclose(money_var_expected, money_var_from_env, atol = 1e-2),
                            f'\nExpected:\n {money_var_expected} \nComputed:\n {money_var_from_env}')
        self.assertTrue(np.allclose(long_var_expected, long_var_from_env, atol = 1e-2),
                            f'\nExpected:\n{long_var_expected} \nComputed:\n {long_var_from_env}')
        self.assertTrue(np.allclose(wallet_expected, wallet_from_env, atol = 1e-2),
                            f'\nExpected:\n{wallet_expected} \nComputed:\n {wallet_from_env}')

    def test_handle_long_position_decreasing_position_by_10percent(self):
        # Start Env for Test
        btc_env = BTCMarket_Env(
                observation_space = (5, 20), # (5, 15)
                action_space = 1,
                start_money = 0,
                trading_fee = 0.001)

        # Init values for testing, based on .reset
        btc_env.windowed_money = [btc_env.start_money]*(btc_env.window_size+1)
        btc_env.inventory = []
        btc_env.money_available = btc_env.start_money # [euros]
        btc_env.long_wallet = [100, 100] # [units of longs, mean_price_bought]
        btc_env.short_wallet = [0, 0] # [units of shorts, mean_price_sold] 
        btc_env.wallet_value = btc_env.start_money + btc_env.long_wallet[0]*btc_env.long_wallet[1] # [euros]
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 1.0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [1.000, 1.000, 0.900, 0.800, 0.700, 0.600, 
                        0.500, 0.400, 0.300, 0.200, 0.100, 0.000, 0.000] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            btc_wallet_variation = action - btc_env.long_position

            new_long_wallet , money_variation, \
                fee_paid =btc_env._handle_long_position(
                    btc_wallet_variation=btc_wallet_variation,
                    btc_price = fixed_priced_for_test,
                    new_position = action,
                    tmp_wallet_value = btc_env.wallet_value,
                    tmp_long_position = btc_env.long_position)
            long_variation_eur = - money_variation - fee_paid
            # Update Vars as in step
            btc_env.money_available += money_variation
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variation)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.00, 0.00, 999.90, 999.70, 999.50, 999.30, 
                        999.10, 998.90, 998.70, 998.50, 998.30, 998.10, 0.00])

        long_var_expected = np.array([0.00, 0.00, -1000.90, -1000.70, -1000.50, 
                        -1000.30, -1000.10, -999.90, -999.70, -999.50, -999.30, -999.10, 0.00])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([[units, 100] for units in [100.00, 100.00, 89.99, 79.98,
                                69.98, 59.98, 49.97, 39.98, 29.98, 19.98, 9.99]], [[0.00, 0.00], [0.00, 0.00]]).flatten()

        self.assertTrue(np.allclose(pos_long_expected, long_pos_env, atol = 1e-2),
                        f'\nExpected:\n {pos_long_expected} \nComputed:\n {long_pos_env} \
                                \nDifference:\n {pos_long_expected, long_pos_env}')

        self.assertTrue(np.allclose(money_var_expected, money_var_from_env, atol = 1e-2),
                        f'\nExpected:\n {money_var_expected} \nComputed:\n {money_var_from_env} \
                                \nDifference:\n {money_var_expected - money_var_from_env}')

        self.assertTrue(np.allclose(long_var_expected, long_var_from_env, atol = 1e-2),
                        f'\nExpected:\n{long_var_expected} \nComputed:\n {long_var_from_env} \
                                \nDifference:\n {long_var_expected - long_var_from_env}')

        self.assertTrue(np.allclose(wallet_expected, wallet_from_env, atol = 1e-2),
                        f'\nExpected:\n{wallet_expected} \nComputed:\n {wallet_from_env} \
                                \nDifference:\n {wallet_expected - wallet_from_env}')
    
    def test_handle_long_position_variations_and_hold_position(self):
        # Start Env for Test
        btc_env = BTCMarket_Env(
                observation_space = (5, 20), # (5, 15)
                action_space = 1,
                start_money = 0,
                trading_fee = 0.001)

        # Init values for testing, based on .reset
        btc_env.windowed_money = [btc_env.start_money]*(btc_env.window_size+1)
        btc_env.inventory = []
        btc_env.money_available = btc_env.start_money # [euros]
        btc_env.long_wallet = [100, 100] # [units of longs, mean_price_bought]
        btc_env.short_wallet = [0, 0] # [units of shorts, mean_price_sold] 
        btc_env.wallet_value = btc_env.start_money + btc_env.long_wallet[0]*btc_env.long_wallet[1] # [euros]
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 1.0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [1.00, 1.00, 0.90, 0.90, 0.80, 0.80, 0.70, 0.70, 
                0.60, 0.60, 0.50, 0.50, 0.40, 0.40, 0.30, 0.30, 0.20, 0.20, 0.10,
                0.10, 0.00, 0.00, 0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.40, 0.40, 
                0.50, 0.50, 0.60, 0.60, 0.70, 0.70, 0.80, 0.80, 0.90, 0.90, 1.00, 1.00] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            btc_wallet_variation = action - btc_env.long_position

            new_long_wallet , money_variation, \
                fee_paid =btc_env._handle_long_position(
                    btc_wallet_variation=btc_wallet_variation,
                    btc_price = fixed_priced_for_test ,
                    new_position = action,
                    tmp_wallet_value = btc_env.wallet_value,
                    tmp_long_position = btc_env.long_position)
            long_variation_eur = - money_variation - fee_paid
            # Update Vars as in step
            btc_env.money_available += money_variation
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variation)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.00, 0.00, 999.90, 0.00, 999.70, 0.00, 999.50, 0.00, 999.30, 0.00, 
                        999.10, 0.00, 998.90, 0.00, 998.70, 0.00, 998.50, 0.00, 998.30, 0.00, 998.10, 0.00, -999.90, 0.00, 
                        -999.70, 0.00, -999.50, 0.00, -999.30, 0.00, -999.10, 0.00, -998.90, 0.00, -998.70, 0.00, -998.50, 
                        0.00, -998.30, 0.00, -998.10, 0.00])

        long_var_expected = np.array([0.00, 0.00, -1000.90, 0.00, -1000.70, 0.00, -1000.50, 0.00, -1000.30, 
                        0.00, -1000.10, 0.00, -999.90, 0.00, -999.70, 0.00, -999.50, 0.00, -999.30, 0.00, -999.10, 0.00, 
                        998.90, 0.00, 998.70, 0.00, 998.50, 0.00, 998.30, 0.00, 998.10, 0.00, 997.90, 0.00, 997.70, 0.00, 
                        997.50, 0.00, 997.30, 0.00, 997.10, 0.00])

        pos_long_expected = np.array(actions_for_test)

        units = np.array([100.00, 100.00, 89.99, 89.99, 79.98, 79.98, 69.98, 69.98, 59.98, 59.98, 49.97, 
                        49.97, 39.98, 39.98, 29.98, 29.98, 19.98, 19.98, 9.99, 9.99, 0.00, 0.00, 9.99, 9.99, 19.98, 19.98, 
                        29.96, 29.96, 39.94, 39.94, 49.93, 49.93, 59.90, 59.90, 69.88, 69.88, 79.86, 79.86, 89.83, 
                        89.83, 99.80, 99.80])

        avg_price = []
        for item in units:
            if item > 0:
                avg_price.append(100)
            else:
                avg_price.append(0.0)

        avg_price = np.array(avg_price)

        wallet_expected = np.array([[u, p] for u,p in zip(units, avg_price)]).flatten()

        self.assertTrue(np.allclose(pos_long_expected, long_pos_env, atol = 1e-2),
                        f'\nExpected:\n {pos_long_expected} \nComputed:\n {long_pos_env} \
                                \nDifference:\n {pos_long_expected, long_pos_env}')

        self.assertTrue(np.allclose(money_var_expected, money_var_from_env, atol = 1e-2),
                        f'\nExpected:\n {money_var_expected} \nComputed:\n {money_var_from_env} \
                                \nDifference:\n {money_var_expected - money_var_from_env}')

        self.assertTrue(np.allclose(long_var_expected, long_var_from_env, atol = 1e-2),
                        f'\nExpected:\n{long_var_expected} \nComputed:\n {long_var_from_env} \
                                \nDifference:\n {long_var_expected - long_var_from_env}')

        self.assertTrue(np.allclose(wallet_expected, wallet_from_env, atol = 1e-2),
                        f'\nExpected:\n{wallet_expected} \nComputed:\n {wallet_from_env} \
                                \nDifference:\n {wallet_expected - wallet_from_env}')

    def test_handle_position_position_greater_than_1(self):
        # Start Env for Test
        btc_env = BTCMarket_Env(
                observation_space = (5, 20), # (5, 15)
                action_space = 1,
                start_money = 10000,
                trading_fee = 0.001)

        # Init values for testing, based on .reset
        btc_env.windowed_money = [btc_env.start_money]*(btc_env.window_size+1)
        btc_env.inventory = []
        btc_env.money_available = btc_env.start_money # [euros]
        btc_env.long_wallet = [0, 0] # [units of longs, mean_price_bought]
        btc_env.short_wallet = [0, 0] # [units of shorts, mean_price_sold] 
        btc_env.wallet_value = btc_env.start_money + btc_env.long_wallet[0]*btc_env.long_wallet[1] # [euros]        
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [0.000, 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 
                0.600, 0.700, 0.800, 0.900, 1.000, 1.1] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            new_long_wallet, short_wallet , money_variation, \
                fee_paid =btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
            long_variation_eur = - money_variation - fee_paid
            # Update Vars as in step
            btc_env.money_available += money_variation
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variation)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([ 0.00, 0.00, -1000.90, -1000.70, -1000.50, 
                        -1000.30, -1000.10, -999.90, -999.70, -999.50, -999.30, -999.10, 0])

        long_var_expected = np.array([0.00, 0.00, 999.90, 999.70, 999.50, 
                                999.30, 999.10, 998.90, 998.70, 998.50, 998.30, 998.10, 0])

        wallet_expected = np.append([[0.0, 0], [0.0, 0]],[[units, 100] for units in [10.00, 20.00, 29.99, 
                        39.98, 49.98, 59.96, 69.95, 79.94, 89.92, 99.90, 99.90]]).flatten()

        pos_long_expected = np.array([0.000, 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 
                0.600, 0.700, 0.800, 0.900, 1.000, 1.0])

        self.assertTrue(np.allclose(pos_long_expected, long_pos_env, atol = 1e-2),
                            f'\nExpected:\n {pos_long_expected} \nComputed:\n {long_pos_env}')
        self.assertTrue(np.allclose(money_var_expected, money_var_from_env, atol = 1e-2),
                            f'\nExpected:\n {money_var_expected} \nComputed:\n {money_var_from_env}')
        self.assertTrue(np.allclose(long_var_expected, long_var_from_env, atol = 1e-2),
                            f'\nExpected:\n{long_var_expected} \nComputed:\n {long_var_from_env}')
        self.assertTrue(np.allclose(wallet_expected, wallet_from_env, atol = 1e-2),
                            f'\nExpected:\n{wallet_expected} \nComputed:\n {wallet_from_env}')

    def test_handle_position_increasing_position_by_10percent(self):
        # Start Env for Test
        btc_env = BTCMarket_Env(
                observation_space = (5, 20), # (5, 15)
                action_space = 1,
                start_money = 10000,
                trading_fee = 0.001)

        # Init values for testing, based on .reset
        btc_env.windowed_money = [btc_env.start_money]*(btc_env.window_size+1)
        btc_env.inventory = []
        btc_env.money_available = btc_env.start_money # [euros]
        btc_env.long_wallet = [0, 0] # [units of longs, mean_price_bought]
        btc_env.short_wallet = [0, 0] # [units of shorts, mean_price_sold] 
        btc_env.wallet_value = btc_env.start_money + btc_env.long_wallet[0]*btc_env.long_wallet[1] # [euros]
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [0.000, 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 
                0.600, 0.700, 0.800, 0.900, 1.000] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            new_long_wallet, short_wallet , money_variation, \
                fee_paid =btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
            long_variation_eur = - money_variation - fee_paid
            # Update Vars as in step
            btc_env.money_available += money_variation
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variation)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([ 0.00, 0.00, -1000.90, -1000.70, -1000.50, 
                        -1000.30, -1000.10, -999.90, -999.70, -999.50, -999.30, -999.10])

        long_var_expected = np.array([0.00, 0.00, 999.90, 999.70, 999.50, 
                                999.30, 999.10, 998.90, 998.70, 998.50, 998.30, 998.10])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([[0.0, 0], [0.0, 0]],[[units, 100] for units in [10.00, 20.00, 29.99, 
                        39.98, 49.98, 59.96, 69.95, 79.94, 89.92, 99.90]]).flatten()

        self.assertTrue(np.allclose(pos_long_expected, long_pos_env, atol = 1e-2),
                            f'\nExpected:\n {pos_long_expected} \nComputed:\n {long_pos_env}')
        self.assertTrue(np.allclose(money_var_expected, money_var_from_env, atol = 1e-2),
                            f'\nExpected:\n {money_var_expected} \nComputed:\n {money_var_from_env}')
        self.assertTrue(np.allclose(long_var_expected, long_var_from_env, atol = 1e-2),
                            f'\nExpected:\n{long_var_expected} \nComputed:\n {long_var_from_env}')
        self.assertTrue(np.allclose(wallet_expected, wallet_from_env, atol = 1e-2),
                            f'\nExpected:\n{wallet_expected} \nComputed:\n {wallet_from_env}')


    def test_handle_position_decreasing_position_by_10percent(self):
        # Start Env for Test
        btc_env = BTCMarket_Env(
                observation_space = (5, 20), # (5, 15)
                action_space = 1,
                start_money = 0,
                trading_fee = 0.001)

        # Init values for testing, based on .reset
        btc_env.windowed_money = [btc_env.start_money]*(btc_env.window_size+1)
        btc_env.inventory = []
        btc_env.money_available = btc_env.start_money # [euros]
        btc_env.long_wallet = [100, 100] # [units of longs, mean_price_bought]
        btc_env.short_wallet = [0, 0] # [units of shorts, mean_price_sold] 
        btc_env.wallet_value = btc_env.start_money + btc_env.long_wallet[0]*btc_env.long_wallet[1] # [euros]
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 1.0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [1.000, 1.000, 0.900, 0.800, 0.700, 0.600, 
                        0.500, 0.400, 0.300, 0.200, 0.100, 0.000, 0.000] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            new_long_wallet, short_wallet , money_variation, \
                fee_paid = btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
            long_variation_eur = - money_variation - fee_paid
            # Update Vars as in step
            btc_env.money_available += money_variation
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variation)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.00, 0.00, 999.90, 999.70, 999.50, 999.30, 
                        999.10, 998.90, 998.70, 998.50, 998.30, 998.10, 0.00])

        long_var_expected = np.array([0.00, 0.00, -1000.90, -1000.70, -1000.50, 
                        -1000.30, -1000.10, -999.90, -999.70, -999.50, -999.30, -999.10, 0.00])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([[units, 100] for units in [100.00, 
                        100.00, 89.99, 79.98, 69.98, 59.98, 49.97, 39.98, 29.98, 
                        19.98, 9.99]], [[0.00, 0.00], [0.00, 0.00]]).flatten()

        self.assertTrue(np.allclose(pos_long_expected, long_pos_env, atol = 1e-2),
                        f'\nExpected:\n {pos_long_expected} \nComputed:\n {long_pos_env} \
                                \nDifference:\n {pos_long_expected, long_pos_env}')

        self.assertTrue(np.allclose(money_var_expected, money_var_from_env, atol = 1e-2),
                        f'\nExpected:\n {money_var_expected} \nComputed:\n {money_var_from_env} \
                                \nDifference:\n {money_var_expected - money_var_from_env}')

        self.assertTrue(np.allclose(long_var_expected, long_var_from_env, atol = 1e-2),
                        f'\nExpected:\n{long_var_expected} \nComputed:\n {long_var_from_env} \
                                \nDifference:\n {long_var_expected - long_var_from_env}')

        self.assertTrue(np.allclose(wallet_expected, wallet_from_env, atol = 1e-2),
                        f'\nExpected:\n{wallet_expected} \nComputed:\n {wallet_from_env} \
                                \nDifference:\n {wallet_expected - wallet_from_env}')

    def test_handle_position_variation_position(self):
        # Start Env for Test
        btc_env = BTCMarket_Env(
                observation_space = (5, 20), # (5, 15)
                action_space = 1,
                start_money = 0,
                trading_fee = 0.001)

        # Init values for testing, based on .reset
        btc_env.windowed_money = [btc_env.start_money]*(btc_env.window_size+1)
        btc_env.inventory = []
        btc_env.money_available = btc_env.start_money # [euros]
        btc_env.long_wallet = [100, 100] # [units of longs, mean_price_bought]
        btc_env.short_wallet = [0, 0] # [units of shorts, mean_price_sold] 
        btc_env.wallet_value = btc_env.start_money + btc_env.long_wallet[0]*btc_env.long_wallet[1] # [euros]
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 1.0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [1.00, 1.00, 0.90, 0.90, 0.80, 0.80, 0.70, 0.70, 
                0.60, 0.60, 0.50, 0.50, 0.40, 0.40, 0.30, 0.30, 0.20, 0.20, 0.10,
                0.10, 0.00, 0.00, 0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.40, 0.40, 
                0.50, 0.50, 0.60, 0.60, 0.70, 0.70, 0.80, 0.80, 0.90, 0.90, 1.00, 1.00] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            new_long_wallet, short_wallet , money_variation, \
                fee_paid =btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
        
            long_variation_eur = - money_variation - fee_paid
            # Update Vars as in step
            btc_env.money_available += money_variation
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variation)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.00, 0.00, 999.90, 0.00, 999.70, 0.00, 999.50, 0.00, 999.30, 0.00, 
                        999.10, 0.00, 998.90, 0.00, 998.70, 0.00, 998.50, 0.00, 998.30, 0.00, 998.10, 0.00, -999.90, 0.00, 
                        -999.70, 0.00, -999.50, 0.00, -999.30, 0.00, -999.10, 0.00, -998.90, 0.00, -998.70, 0.00, -998.50, 
                        0.00, -998.30, 0.00, -998.10, 0.00])

        long_var_expected = np.array([0.00, 0.00, -1000.90, 0.00, -1000.70, 0.00, -1000.50, 0.00, -1000.30, 
                        0.00, -1000.10, 0.00, -999.90, 0.00, -999.70, 0.00, -999.50, 0.00, -999.30, 0.00, -999.10, 0.00, 
                        998.90, 0.00, 998.70, 0.00, 998.50, 0.00, 998.30, 0.00, 998.10, 0.00, 997.90, 0.00, 997.70, 0.00, 
                        997.50, 0.00, 997.30, 0.00, 997.10, 0.00])

        pos_long_expected = np.array(actions_for_test)

        units = np.array([100.00, 100.00, 89.99, 89.99, 79.98, 79.98, 69.98, 69.98, 59.98, 59.98, 49.97, 
                        49.97, 39.98, 39.98, 29.98, 29.98, 19.98, 19.98, 9.99, 9.99, 0.00, 0.00, 9.99, 9.99, 19.98, 19.98, 
                        29.96, 29.96, 39.94, 39.94, 49.93, 49.93, 59.90, 59.90, 69.88, 69.88, 79.86, 79.86, 89.83, 
                        89.83, 99.80, 99.80])

        avg_price = []
        for item in units:
            if item > 0:
                avg_price.append(100)
            else:
                avg_price.append(0.0)

        avg_price = np.array(avg_price)

        wallet_expected = np.array([[u, p] for u,p in zip(units, avg_price)]).flatten()

        self.assertTrue(np.allclose(pos_long_expected, long_pos_env, atol = 1e-2),
                        f'\nExpected:\n {pos_long_expected} \nComputed:\n {long_pos_env} \
                                \nDifference:\n {pos_long_expected, long_pos_env}')

        self.assertTrue(np.allclose(money_var_expected, money_var_from_env, atol = 1e-2),
                        f'\nExpected:\n {money_var_expected} \nComputed:\n {money_var_from_env} \
                                \nDifference:\n {money_var_expected - money_var_from_env}')

        self.assertTrue(np.allclose(long_var_expected, long_var_from_env, atol = 1e-2),
                        f'\nExpected:\n{long_var_expected} \nComputed:\n {long_var_from_env} \
                                \nDifference:\n {long_var_expected - long_var_from_env}')

        self.assertTrue(np.allclose(wallet_expected, wallet_from_env, atol = 1e-2),
                        f'\nExpected:\n{wallet_expected} \nComputed:\n {wallet_from_env} \
                                \nDifference:\n {wallet_expected - wallet_from_env}')


if __name__ == '__main__':
    unittest.main()
    # tmp = TestEnv()
    # tmp.test_handle_long_position_variations_and_hold_position()