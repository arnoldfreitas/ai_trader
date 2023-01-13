
from curses import window
from datetime import datetime
import os
import json
import math
import random
import shutil
import traceback
from xml.dom.expatbuilder import parseString
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas_datareader as data_reader

from tqdm import tqdm_notebook, tqdm
from collections import deque

class AI_Trader():
    '''
    Class Encapsulating all aspects on the RL training process:
        Agent; Env; Trainer; Reward; Model

    Attributes
    -----------
    state_size: int
        Number of Variables in the state space.
    window_size: int
        Time window length for states and actions. #TODO: I assumed, Check if is true.
    data_path
        Path to data folder with source-CSV, training history and images.
    action_space:
        Number of variables on action space.
    cols
        Name of Columns on CSV to be used on train.
    data: pd.DataFrame

    memory
        Memory holder of states, actions, rewards.
    inventory
        Holds all on-going trade positions.
    model_name
        Name of model
    start_money
        Initial amount of money available for agent.
    gamma
        Discount Factor for predicted futures reward Q(state,action)
    epsilon
        Epsilon Greedy Value. Define percentage of random steps while sampling train data.
    epsilon_final
        Final Epsilon Greedy Value.
    epsilon_decay
        Decay Value for Epsilon Greedy.
    model
        TensorFlow Model for RL-Agent
    '''
  
    def __init__(self, state_size,window_size,start_money=30, action_space=4, model_name="AITrader",data_path='./data',load_model=True):

        self.state_size=state_size
        self.window_size=window_size
        self.data_path=data_path
        self.action_space = action_space
        self.cols=['date','action','state','profit']
        self.data=pd.DataFrame(columns=self.cols)
        self.memory = deque() # experience replay
        self.inventory = [] # postions
        self.model_name = model_name
        self.start_money=start_money

        self.gamma = 0.95
        self.epsilon = 1.0 # exploration or not 1 is full random, start with 1 
        self.epsilon_final = 0.01 # final epsilon
        self.epsilon_decay = 0.995
        if load_model:
            try:
                self.load_model()
            except:
                print('Model loading failed:')
                print(traceback.print_exc())
                self.model=self.model_builder()
        else:
            self.model=self.model_builder()

    def load_model(self):
        '''
        Load TensorFlow Model from h5 file. 

        Note
        ----
        Set self.epsilon to 0.5
        '''
        epi_list=[]
        date_list=[] 
        for file in os.listdir(self.data_path+"/Bot/models"):
            if '.h5' in file:
                date_list.append(file.split('.')[0].split('_')[2])
                epi_list.append(int(file.split('.')[0].split('_')[3]))
        load_epi=max(epi_list)
        load_date=date_list[epi_list.index(load_epi)] 
        self.model=load_model(self.data_path+"/Bot/models/ai_trade_{}_{}.h5".format(load_date,load_epi))
        self.epsilon=0.5
        print("model: ai_trade_{}_{} loaded. Eplison set to {}.".format(load_date,load_epi,self.epsilon))

    def model_builder(self):
        '''
        Make Tensorflow MLP-Model with architecture:
        
        Returns
        -------
        TensorFlow Model
        '''
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

    def trade(self, state):
        '''
        Performs Trade action from an given state. Uses epsilon greedy method or model.predict to define action.
        
        Returns
        -------
        Computed actions
        '''
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
      
        actions = self.model.predict(tf.reshape(tf.convert_to_tensor(state[0],dtype=np.float32),shape=(1,self.state_size)),verbose = 0)
        return np.argmax(actions)
    
    def batch_train(self, batch_size):
        '''
        Fit model to batch of batch_size

        Notes
        -----
        Following DQN Algorithm the reward decay (gamma) is used in this function to define future rewards.

        Parameters
        ----------
        batch_size : int
            Size of Batch to be use on model.fit
        
        Returns
        -------
        reults : 
            Tensorflow Training History
        '''
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state,verbose = 0)[0]) 

            target = self.model.predict(state,verbose = 0)
            target[0][action] = reward

            result=self.model.fit(state, target, epochs=5, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
        
        return result
    
    def state_creator(self,data,windowed_money, timestep, window_size):
        '''
        Generate state from data 

        Notes
        -----
        Apply Sigmoid Function to States. #TODO: Why not directly use the states values? --> normalize data, model shouldnt be affected by the scales of the measures.

        Parameters
        ----------
        data : 

        windowed_money: 
            Money on wallet for each timestep in time-window
        timestep: int
            Intial Timestep for window.
        window_size: int
           Time window length for states and actions.
        
        Returns
        -------
        Computed States within time window
        #TODO: Define Shape --> NumPy Array with shape: window_size * 5
        '''
  
        starting_id = timestep - window_size
        
        if starting_id >= 0:
            windowed_close_data = data['close'].values[starting_id:timestep+1]
            windowed_hist_data = data['histogram'].values[starting_id:timestep+1]
            windowed_ema_data = data['50ema'].values[starting_id:timestep+1]
            windowed_rsi_data = data['rsi14'].values[starting_id:timestep+1]
        else:
            windowed_close_data = [data['close'].values[0]]*abs(starting_id) + list(data['close'].values[0:timestep+1])
            windowed_hist_data = [data['histogram'].values[0]]*abs(starting_id) + list(data['histogram'].values[0:timestep+1])
            windowed_ema_data = [data['50ema'].values[0]]*abs(starting_id) + list(data['50ema'].values[0:timestep+1])
            windowed_rsi_data = [data['rsi14'].values[0]]*abs(starting_id) + list(data['rsi14'].values[0:timestep+1])
            
        state = []
        for i in range(window_size):
            state.append(self.sigmoid(windowed_close_data[i+1] - windowed_close_data[i]))
            state.append(self.sigmoid(windowed_hist_data[i]))
            state.append(self.sigmoid(windowed_ema_data[i+1] - windowed_ema_data[i]))
            state.append(self.sigmoid(windowed_money[i+1] - windowed_money[i]))
            state.append(windowed_rsi_data[i]/100)
        #state.append(money)

        return np.array([np.nan_to_num(state)])

    def sigmoid(self,x):
        try:
            result = math.exp(-x)
        except OverflowError:
            result = math.inf
        return 1 /(1 + result)

    @staticmethod 
    def getrandomSample(data,period):
        start=random.randint(0,len(data)-period)
        return data.iloc[start:start+period,:] 

    def get_reward_money(self,start_money, actual_price,money,past_profit,fee_p=0,action=None):
        '''
        Reward Function
        Returns
        -------
        reward
            Reward Value
        act_val
            Cash outstanding after executing action
        fee
            Fees paid by agent
        '''
        act_val= 0
        for invest,buy_in in self.inventory:
            pos_yield=(actual_price-buy_in)/buy_in
            act_val+=invest+(invest*pos_yield)
        fee=act_val*fee_p
        act_val-=fee
        act_val+=money
        reward=act_val-start_money+past_profit
        if action is not None:
            if action == 3:
                reward=reward
        return reward,act_val,fee

    def dataset_loader(self,symbol):
        '''
        Load dataset using pandas data reader.

        Parameters
        ----------
        symbol:
            #TODO: What is that exactly? --> name of asset (in this case)
            DataReader creates df objects by using data sources from the internet. Popular for realtime stock price datasets.

        Returns
        -------
        Dataset Columns named Closed 
        '''
        dataset = data_reader.DataReader(symbol, data_source="binance")
        
        start_date = str(dataset.index[0]).split()[0]
        end_date = str(dataset.index[1]).split()[0]
        
        close = dataset['Close']
        
        return close

    def stock_price_format(self,n):
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
    
    def save_data(self,action_data,epi_data,episode,train_data,epi_dataFrame,overwrite=False,save_model=True):
        '''
        Save Episode Data to csv

        Parameters:
        action_data

        epi_data

        episode

        train_data

        epi_dataFrame

        overwrite=False

        save_model=True
        '''
        save_str=datetime.now().strftime('%Y%m%d')
        save_path=self.data_path+'/Bot/RL_Bot/'+save_str
        index=0
        if overwrite and os.path.exists(save_path):
            shutil.rmtree(save_path)
        #while os.path.exists(save_path):
        #    save_path=save_path+'_'+str(index)
        os.makedirs(save_path,exist_ok=True)
        epi=action_data['episode'].unique()[0]
        action_data.to_csv(save_path+f"/action_data_e{epi}.csv")
        with open(save_path+"/train_data.json","w") as out_file:
            json.dump(train_data,out_file)
        if save_model:
            self.model.save(self.data_path+"/Bot/models/ai_trade_{}_{}.h5".format(save_str,episode))
        tmp=pd.DataFrame(epi_data,columns=self.epi_cols)
        epi_dataFrame=pd.concat([epi_dataFrame,tmp])
        epi_dataFrame.to_csv(save_path+"/Epi_data.csv")
        print('Data saved')

    def train(self,episodes,window_size,data_in,batch_size):
        '''
        Main Train function.

        Notes
        -----
        More specific documentation with in-line comments on code

        Parameters
        ----------
        episodes:
            Number of episodes to be played.
        window_size: int
           Time window length for states and actions.
        data_in

        batch_size: int
         Size of Batch for fitting model
        '''
        # Init Parameters for Train Algorithm
        data_samples_in = len(data_in) - 1
        #data_samples =int(data_samples // 1000)
        self.epi_cols=['episode','#buy_actions','#sell_actions','money','fee','profit','epsilon']
        #action_data=pd.DataFrame(columns=['episode','run','date','action','state','money_free','money_fiktiv','invest','fee','reward','profit'])
        epi_dataFrame=pd.DataFrame(columns=self.epi_cols)
        runs = 10 # Max number of runs on each episode
        period=2*24*14 # [hours?] 1h-> 1Tag -> 2 wochen TODO: Why the 2 in front? --> Wir haben halbstündige daten. Dh 2 Wochen.
        invest=0 # Invested Value
        start_money=200 # Initial Money available on wallet
        windowed_money=[start_money]*(window_size+1) # money on wallet for each timestep in time-window
        fee_p=0.001 # Fee paid for trade, as percentage of trade value
        train_cnt=0 # Counter for train iterations
        epi_data=[]
        # Loop over every episode
        for episode in range(1, episodes + 1):
            print("Episode: {}/{}".format(episode, episodes))
            if episode % 10 == 0: # Increase Epsilon every 10 episodes
                self.epsilon+=0.5
                print(f'on Episode {episode} set Eplison to {self.epsilon} to find global minimum')
            run_profit=0.0 # Init Profit on episode
            # Init dataframe for runs
            action_data=pd.DataFrame(columns=['episode','run','timestep','date','action','state','money_free','money_fiktiv','invest','fee','reward','profit'])
            # Loop inside one episode over number runs 
            for run in range(1,runs+1):
                print("Episode: {}/{} || Run {}/{}".format(episode, episodes,run,runs))
                if run % 5 == 0: # Increase epsilon every 5 runs
                    self.epsilon+=(0.5 -(run/runs)*0)
                    print(f'on Run {run} set Eplison to {self.epsilon} to find global minimum')
                data=self.getrandomSample(data_in,period)
                #action_data=pd.DataFrame(columns=['episode','run','timestep','date','action','state','money_free','money_fiktiv','invest','fee','reward','profit'])
                # Init params to be used on run
                data_samples=len(data)-1 
                total_profit = 0.0 # Profit on run
                self.inventory = [] # Wallet inventory
                train_data={}
                buy_cnt=0 # Counter for buys
                fees=0 # Fees expenses on this run
                sell_cnt=0  # Counter for buys
                money_free=start_money # Init money available
                money_fiktiv=money_free
                windowed_money=[money_fiktiv]*(window_size+1)
                invest=0
                state = self.state_creator(data,windowed_money, 0, window_size) # Gatter state from data
                for t in tqdm(range(data_samples)):
                    action = self.trade(state) # Compute Action given state
                    next_state = self.state_creator(data,windowed_money, t+1, window_size) # Gatter State after action
                    reward = 0 # Init Reward
                    # Evaluate if money is over and inventory is empty, thus has lost all money.
                    if money_free <=1 and len(self.inventory)<=0:
                        epi_data.append([episode,buy_cnt,sell_cnt,round(money_fiktiv,2),round(fees,2),round(total_profit,2),self.epsilon])
                        break
                    # Conduct evaluations and compute reward depending on which action was taken.
                    if action == 1 and money_free>20: #Buying 50%
                        buy_cnt+=1 
                        tmp_invest=money_free*0.5
                        money_free-=tmp_invest
                        fee=tmp_invest*fee_p
                        tmp_invest-=fee
                        invest+=tmp_invest
                        self.inventory.append((tmp_invest,data['close'].values[t]))
                        fees+=fee
                        reward,money_fiktiv,_ = self.get_reward_money(start_money,data['close'].values[t],money_free,total_profit,action=action)
                        action_data=pd.concat([action_data,pd.DataFrame([[episode,run,t,data['date'].values[t],'buy_50',list(state[0]),round(money_free,2),round(money_fiktiv,2),round(invest,2),round(fee,2),round(reward,2),round(total_profit,4)]],columns=action_data.columns)])
                        #print("AI Trader bought: ", self.stock_price_format(data['close'].values[t]))
                    elif action == 2 and money_free>0: #Buying 100%
                        buy_cnt+=1
                        tmp_invest=money_free
                        money_free-=tmp_invest
                        fee=tmp_invest*fee_p
                        tmp_invest-=fee
                        fees+=fee
                        invest+=tmp_invest
                        self.inventory.append((tmp_invest,data['close'].values[t]))
                        reward,money_fiktiv,_ = self.get_reward_money(start_money,data['close'].values[t],money_free,total_profit,action=action)
                        action_data=pd.concat([action_data,pd.DataFrame([[episode,run,t,data['date'].values[t],'buy_100',list(state[0]),round(money_free,2),round(money_fiktiv,2),round(invest,2),round(fee,2),round(reward,2),round(total_profit,4)]],columns=action_data.columns)])
                        #print("AI Trader bought: ", self.stock_price_format(data['close'].values[t]))
                    elif action == 3 and len(self.inventory) > 0: #Selling
                        sell_cnt+=1
                        money_before=money_free
                        reward,money_free,fee = self.get_reward_money(start_money,data['close'].values[t],money_free,total_profit,fee_p,action=action)
                        fees+=fee
                        money_fiktiv=money_free
                        sell_profit=(money_free-money_before-sum([x[0] for x  in self.inventory]))
                        total_profit += sell_profit
                        invest=0
                        action_data=pd.concat([action_data,pd.DataFrame([[episode,run,t,data['date'].values[t],'sell',list(state[0]),round(money_free,2),round(money_fiktiv,2),round(invest,2),round(fee,2),round(reward,2),round(total_profit,4)]],columns=action_data.columns)])
                        self.inventory=[]
                        
                    else:
                        reward,money_fiktiv,_ = self.get_reward_money(start_money,data['close'].values[t],money_free,total_profit,action=action)
                        action_data=pd.concat([action_data,pd.DataFrame([[episode,run,t,data['date'].values[t],'hold',list(state[0]),round(money_free,2),round(money_fiktiv,2),round(invest,2),0.0,round(reward,2),round(total_profit,4)]],columns=action_data.columns)])
                        #reward = max(data['close'].values[t] - buy_price.0)
                        #print("AI Trader sold: ", self.stock_price_format(data['close'].values[t]), " Profit: " + self.stock_price_format(data['close'].values[t] - buy_price) )
                    # Identify if all samples were used. 
                    if t == data_samples - 1:
                        done = True
                    else:
                        done = False
                        self.memory.append((state, action, reward, next_state, done))
                        windowed_money.pop(0)
                        windowed_money.append(money_fiktiv)
                        state = next_state
                    # Finish run if all data sample has been used
                    if done:
                        epi_data.append([episode,buy_cnt,sell_cnt,round(money_fiktiv,2),round(fees,2),round(total_profit,2),self.epsilon])
                        print("/n ########################")
                        print(f"TOTAL PROFIT: {total_profit:.2f}")
                        print(f'#buys {buy_cnt}, #sells {sell_cnt}')
                        print("########################")
                        break
                    # Conduct Training in batch
                    if len(self.memory) > batch_size:
                        res=self.batch_train(batch_size)
                        epi_string='Epi_'+str(episode)
                        if epi_string not in train_data:
                            train_data.update({epi_string:{'loss':[res.history['loss']],'#trains':train_cnt,'epsilon':[self.epsilon]}})
                        else:
                            train_data[epi_string]['loss'].append(res.history['loss'])
                            train_data[epi_string]['#trains']=train_cnt
                            train_data[epi_string]['epsilon'].append(self.epsilon)  
                        train_cnt+=1
                    # Checkpoint data
                    if t >=100 and t % 100 == 0:
                        self.save_data(action_data,epi_data,episode,train_data,epi_dataFrame,save_model=False)
                        #print(f'episode {episode}, sample ({t}/{len(data)-1}).Profit {total_profit:.2f} || money: {money:.2f} || invest: {invest:.2f} || #buys {buy_cnt}, #sells {sell_cnt}')
                        print(f'episode {episode}, run ({run}/{runs}) sample ({t}/{len(data)-1}).Profit {total_profit:.2f} || money fiktiv: {(money_fiktiv):.2f} || #buys {buy_cnt}, #sells {sell_cnt}')
                run_profit+=total_profit
                print(f'episode {episode}, finished run ({run}/{runs}). Run Profit {run_profit:.2f} || money fiktiv: {(money_fiktiv):.2f} || #buys {buy_cnt}, #sells {sell_cnt}')
            print(f'episode {episode}/{episodes}. Profit {total_profit:2f} || money fiktiv: {money_fiktiv:.2f} ||  invest: {invest:.2f} || #buys {buy_cnt}, #sells {sell_cnt}')
            self.save_data(action_data,epi_data,episode,train_data,epi_dataFrame)
    
def loadData(onefile=False,asset=None):
    out=[]
    for file in os.listdir('./data'):
        if asset is not None and asset not in file:
            continue
        if 'histData_dt1800.0s' in file:
            out.append(pd.read_csv('./data/'+file))
    if onefile:
        out_df=pd.DataFrame(columns=out[0].columns)
        for item in out:
            out_df=pd.concat([out_df,item])
        return out_df
    else:
        return out


if __name__ == "__main__":
    # TODO: where the package utils come from?
    # data=utils.loadData(onefile=False,asset='BTC')[0] # hier csv daten laden
    data=loadData(onefile=False,asset='BTC')[0] # hier csv daten laden
    window_size = 20
    state_size = 100 # 4 stes ( close, hist,rsi,ema) und money
    episodes = 1000

    batch_size = 32
    trader = AI_Trader(state_size=state_size,window_size=window_size,load_model=True)
    trader.model.summary()
    trader.train(episodes,window_size,data,batch_size)