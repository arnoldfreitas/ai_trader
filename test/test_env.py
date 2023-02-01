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
        btc_env.wallet_value = btc_env.start_money # [euros]:
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
            btc_wallet_variaton = action - btc_env.long_position

            new_long_wallet , money_variaton, \
                long_variation_eur =btc_env._handle_long_position(
                    btc_wallet_variaton=btc_wallet_variaton,
                    btc_price = fixed_priced_for_test )
            # Update Vars as in step
            btc_env.money_available += money_variaton
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variaton)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.000, 0.000, -999.999, -1000.799, -1000.600,
                -1000.400, -1000.200, -1000.000, -999.800, -999.600, -999.400, -999.201])

        long_var_expected = np.array([0.000, 0.000, 999.000, 999.799, 999.600, 
                    999.400, 999.201, 999.001, 998.801, 998.602, 998.402, 998.202])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([[0.0, 0], [0.0, 0]],[[units, 100] for units in [9.990, 19.988, 29.984, 39.978, 49.970, 59.960, 69.948, 
                79.934, 89.918, 99.900]]).flatten()

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
        btc_env.wallet_value = btc_env.start_money # [euros]:
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
            btc_wallet_variaton = action - btc_env.long_position

            new_long_wallet , money_variaton, \
                long_variation_eur =btc_env._handle_long_position(
                    btc_wallet_variaton=btc_wallet_variaton,
                    btc_price = fixed_priced_for_test )
            # Update Vars as in step
            btc_env.money_available += money_variaton
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variaton)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.000, 0.000, 998.001, 999.797, 999.600, 
                    999.400, 999.200, 999.000, 998.800, 998.600, 998.401, 998.201, 0.998])

        long_var_expected = np.array([0.000, 0.000, -999.000, -1000.797, -1000.601, 
                -1000.401, -1000.200, -1000.000, -999.800, -999.600, -999.400, 
                -999.200, -0.999])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([100, 100],[[units, 100] for units in [100.000, 
                90.010, 80.002, 69.996, 59.992, 49.990, 39.990, 29.992, 
                19.996, 10.002, 0.010, 0.000]]).flatten()

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
        btc_env.wallet_value = btc_env.start_money # [euros]:
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 1.0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [1.000, 1.000, 0.900, 0.900, 0.800, 0.800, 0.800, 0.700, 
                0.700, 0.600, 0.500, 0.500, 0.300, 0.300, 0.400, 0.400, 0.500, 0.500, 
                0.500, 0.700, 0.700, 0.900, 0.800, 0.300, 1.000] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            btc_wallet_variaton = action - btc_env.long_position

            new_long_wallet , money_variaton, \
                long_variation_eur =btc_env._handle_long_position(
                    btc_wallet_variaton=btc_wallet_variaton,
                    btc_price = fixed_priced_for_test )
            # Update Vars as in step
            btc_env.money_available += money_variaton
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variaton)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.000, 0.000, 998.001, 1.895, 997.905, 1.795, 
                0.003, 997.801, 1.696, 997.704, 999.197, 1.498, 1995.006, 2.593, 
                -999.295, -0.600, -999.199, -0.500, 0.000, -1998.198, -0.601, 
                -1997.799, 996.503, 4984.817, -6980.216])

        long_var_expected = np.array([0.000, 0.000, -999.000, -1.897, -998.904, 
            -1.797, -0.003, -998.800, -1.697, -998.703, -1000.198, -1.500, -1997.003, 
            -2.596, 998.297, 0.599, 998.201, 0.500, 0.000, 1996.202, 0.600, 1995.803, 
            -997.501, -4989.807, 6973.242])

        pos_long_expected = np.array(actions_for_test)

        units = np.array([100.000, 100.000, 90.010, 89.991, 80.002, 79.984, 79.984, 
        69.996, 69.979, 59.992, 49.990, 49.975, 30.005, 29.979, 39.962, 39.968, 
        49.950, 49.955, 49.955, 69.917, 69.923, 89.881, 79.906, 30.008, 99.740])

        avg_price = np.array([100]*len(units))

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
        btc_env.wallet_value = btc_env.start_money # [euros]:
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
            new_long_wallet , money_variaton, \
                long_variation_eur =btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
            # Update Vars as in step
            btc_env.money_available += money_variaton
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variaton)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.000, 0.000, -999.999, -1000.799, -1000.600,
                -1000.400, -1000.200, -1000.000, -999.800, -999.600, -999.400, -999.201, 0])

        long_var_expected = np.array([0.000, 0.000, 999.000, 999.799, 999.600, 
                    999.400, 999.201, 999.001, 998.801, 998.602, 998.402, 998.202, 0])

        pos_long_expected = np.array([0.000, 0.000, 0.100, 0.200, 0.300, 0.400, 0.500, 
                0.600, 0.700, 0.800, 0.900, 1.000, 1.0])

        wallet_expected = np.append([[0.0, 0], [0.0, 0]],[[units, 100] for units in [9.990, 19.988, 29.984, 39.978, 49.970, 59.960, 69.948, 
                79.934, 89.918, 99.900, 99.900]]).flatten()

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
        btc_env.wallet_value = btc_env.start_money # [euros]:
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
            new_long_wallet , money_variaton, \
                long_variation_eur =btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
            # Update Vars as in step
            btc_env.money_available += money_variaton
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variaton)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.000, 0.000, -999.999, -1000.799, -1000.600,
                -1000.400, -1000.200, -1000.000, -999.800, -999.600, -999.400, -999.201])

        long_var_expected = np.array([0.000, 0.000, 999.000, 999.799, 999.600, 
                    999.400, 999.201, 999.001, 998.801, 998.602, 998.402, 998.202])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([[0.0, 0], [0.0, 0]],[[units, 100] for units in [9.990, 19.988, 29.984, 39.978, 49.970, 59.960, 69.948, 
                79.934, 89.918, 99.900]]).flatten()

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
        btc_env.wallet_value = btc_env.start_money # [euros]:
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
            new_long_wallet , money_variaton, \
                long_variation_eur =btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
            # Update Vars as in step
            btc_env.money_available += money_variaton
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variaton)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.000, 0.000, 998.001, 999.797, 999.600, 
                    999.400, 999.200, 999.000, 998.800, 998.600, 998.401, 998.201, 0.998])

        long_var_expected = np.array([0.000, 0.000, -999.000, -1000.797, -1000.601, 
                -1000.401, -1000.200, -1000.000, -999.800, -999.600, -999.400, 
                -999.200, -0.999])

        pos_long_expected = np.array(actions_for_test)

        wallet_expected = np.append([100, 100],[[units, 100] for units in [100.000, 
                90.010, 80.002, 69.996, 59.992, 49.990, 39.990, 29.992, 
                19.996, 10.002, 0.010, 0.000]]).flatten()

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

    def test_handle_position_variaton_position(self):
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
        btc_env.wallet_value = btc_env.start_money # [euros]:
        btc_env.money_fiktiv = np.array([btc_env.wallet_value]) # [euros]
        btc_env.long_position = 1.0 # [%]
        btc_env.short_position = 0 # [%]

        btc_env.buy_long_count = 0
        btc_env.sell_long_count = 0
        btc_env.buy_short_count = 0
        btc_env.sell_short_count = 0
        
        actions_for_test = [1.000, 1.000, 0.900, 0.900, 0.800, 0.800, 0.800, 0.700, 
                0.700, 0.600, 0.500, 0.500, 0.300, 0.300, 0.400, 0.400, 0.500, 0.500, 
                0.500, 0.700, 0.700, 0.900, 0.800, 0.300, 1.000] 
        fixed_priced_for_test = 100

        
        money_var_from_env = np.array([])
        long_var_from_env = np.array([])
        wallet_from_env = np.array([])
        long_pos_env = np.array([])

        for action in actions_for_test:
            # Compute new Position
            new_long_wallet , money_variaton, \
                long_variation_eur =btc_env._handle_position(
                    new_position=action,
                    btc_price = fixed_priced_for_test )
            # Update Vars as in step
            btc_env.money_available += money_variaton
            btc_env.long_wallet = new_long_wallet
            btc_env.wallet_value = btc_env.money_available + new_long_wallet[0]*new_long_wallet[1]
            btc_env.long_position = new_long_wallet[0]*new_long_wallet[1]/ btc_env.wallet_value 

            # Save Computed values for Assertation
            money_var_from_env = np.append(money_var_from_env, money_variaton)
            long_var_from_env = np.append(long_var_from_env, long_variation_eur)
            wallet_from_env = np.append(wallet_from_env, new_long_wallet)
            long_pos_env = np.append(long_pos_env, btc_env.long_position)

        money_var_expected = np.array([0.000, 0.000, 998.001, 1.895, 997.905, 1.795, 
                0.003, 997.801, 1.696, 997.704, 999.197, 1.498, 1995.006, 2.593, 
                -999.295, -0.600, -999.199, -0.500, 0.000, -1998.198, -0.601, 
                -1997.799, 996.503, 4984.817, -6980.216])

        long_var_expected = np.array([0.000, 0.000, -999.000, -1.897, -998.904, 
            -1.797, -0.003, -998.800, -1.697, -998.703, -1000.198, -1.500, -1997.003, 
            -2.596, 998.297, 0.599, 998.201, 0.500, 0.000, 1996.202, 0.600, 1995.803, 
            -997.501, -4989.807, 6973.242])

        pos_long_expected = np.array(actions_for_test)

        units = np.array([100.000, 100.000, 90.010, 89.991, 80.002, 79.984, 79.984, 
        69.996, 69.979, 59.992, 49.990, 49.975, 30.005, 29.979, 39.962, 39.968, 
        49.950, 49.955, 49.955, 69.917, 69.923, 89.881, 79.906, 30.008, 99.740])

        avg_price = np.array([100]*len(units))

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
    # test_handle_long_position_increasing_position_by_10percent()