import os
import json
import shutil

import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import numpy as np
from mplfinance.original_flavor import candlestick2_ohlc
from tqdm import tqdm

data_path=r'Bot/data'
#epi_data_path=r'Bot/data/Epi_data.csv'
#train_data_path=r'Bot/data/train_data.json'
coin_data_path='Bot/data/BTC_histData_dt1800.0s_20220825_0629.csv'

save_figs_path=f'Bot/data/pics'

run_interval=671

def vis_action_data(save_figs):
    file_list=[]
    for folder in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path,folder)) or 'pics' in folder:
            continue
        for file in os.listdir(os.path.join(data_path,folder)):
            seen=False
            for in_file in file_list:
                if file in in_file:
                    seen =True
                    if os.path.getsize(os.path.join(data_path,folder,file))>os.path.getsize(in_file):
                        idx=file_list.index(in_file)
                        file_list.pop(idx)
                        file_list.insert(idx,os.path.join(data_path,folder,file))
                        break
            if not seen:
                file_list.append(os.path.join(data_path,folder,file))

    folder_list=[x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,x)) and not 'pics' in x]
    epi_csv_data=pd.read_csv(os.path.join(data_path,max(folder_list),'Epi_data.csv'))
    fig_cnt=0
    coins_data=pd.read_csv(coin_data_path)
    money_list_all=[]
    profit_list_all=[]
    epi_s=epi_csv_data['episode'].unique()
    for epi in tqdm(epi_s):
        epi_data=epi_csv_data.loc[epi_csv_data['episode']==epi]
        found =False
        for file in file_list:
            if f'action_data_e{epi}' in file:
                action_data=pd.read_csv(file)
                found=True
        if not found:
            continue
        if save_figs:
            if not os.path.exists(os.path.join(save_figs_path, str(epi))):
                os.makedirs(os.path.join(save_figs_path, str(epi)))
        if not 'run' in epi_data:
            run=0
            run_list=[0]*len(epi_data)
            for i in range(0,len(epi_data),run_interval):
                run_list[i:i+run_interval]=[run]*len(run_list[i:i+run_interval])
                run+=1
            epi_data['run']=run_list
        money_list=[]
        profit_list=[]
        fig_1, axs_1 = plt.subplots(figsize=(12, 4))
        fig_2, axs_2 = plt.subplots(figsize=(12, 4))
        fig_3, axs_3 = plt.subplots(figsize=(12, 4))
        for run,run_data in action_data.groupby('run'):
            run_coin_data=coins_data.loc[(coins_data['date']>=run_data['date'].values[0]) & (coins_data['date']<=run_data['date'].values[-1])]
            money_list.append(run_data['money_fiktiv'].values[-1])
            profit_list.append(run_data['profit'].values[-1])
            axs_1.plot(list(range(0,len(run_data))),run_data['profit'], label=f'Run {run}')
            axs_1.legend()
            fig_1.suptitle(f'Realisierter Profit über die Runs für Episode {epi}')
            axs_2.plot(list(range(0,len(run_data))),run_data['money_fiktiv'], label=f'Run {run}')
            axs_2.legend()
            fig_2.suptitle(f'Wertentwicklung über die Runs für Episode {epi}')
            axs_3.plot(list(range(0,len(run_data))),run_data['reward'], label=f'Run {run}')
            axs_3.legend()
            fig_3.suptitle(f'Rewards über die Runs für Episode {epi}')
            visu_buys_sells_in_run(epi,run,run_coin_data,run_data,save_figs=True)
        plot_value_profit(epi,money_list, profit_list, save_figs)
        money_list_all+=money_list
        profit_list_all+=profit_list
        if save_figs:
            fig_1.savefig(os.path.join(save_figs_path,str(epi), f'fig_e{epi}_profit.png'))
            fig_2.savefig(os.path.join(save_figs_path,str(epi),f'fig_e{epi}_wert.png'))
            fig_3.savefig(os.path.join(save_figs_path,str(epi),f'fig_e{epi}_reward.png'))
    plot_value_profit('all',money_list_all, profit_list_all, save_figs)
    if not save_figs:
        plt.show()
    print('done')

def plot_value_profit(epi,money_list,profit_list,save_figs):
    fig_3, axs_3 = plt.subplots(figsize=(12, 4))
    axs_3.plot(list(range(0, len(money_list))), money_list, label='Wert',color='blue')
    axs_3.plot(list(range(0, len(money_list))), np.cumsum([x -200 for x in money_list]), '--', label=' kumulierter Wert',color='blue')
    axs_3.plot(list(range(0, len(profit_list))), profit_list, label='Profit',color='orange')
    axs_3.plot(list(range(0, len(money_list))), np.cumsum(profit_list), '--', label=' kumulierter Profit',color='orange')
    axs_3.legend()
    if epi == 'all':
        fig_3.suptitle(f'Wertentwicklung /Profit am Ende eines Runs über alle Episoden')
    else:
        fig_3.suptitle(f'Wertentwicklung /Profit am Ende eines Runs über Episode {epi}')
    for i, txt in enumerate(money_list):
        if money_list[i] < 200:
            c = 'red'
        elif money_list[i] >= 200 and money_list[i] < 210:
            c = 'black'
        else:
            c = 'green'
        axs_3.annotate(txt, (i, money_list[i]), color=c)

        if profit_list[i] < 0:
            c = 'red'
        elif profit_list[i] >= 0 and money_list[i] < 10:
            c = 'black'
        else:
            c = 'green'
        axs_3.annotate(f'{profit_list[i]:.2f}', (i, profit_list[i]), color=c)
    if save_figs:
        if epi=='all':
            fig_3.savefig(os.path.join(save_figs_path, f'Wert_profit_all.png'))
        else:
            fig_3.savefig(os.path.join(save_figs_path,str(epi), f'Wert_profit_e{epi}.png'))


def cleanPics():
    shutil.rmtree(save_figs_path)
    os.makedirs(save_figs_path,exist_ok=True)

def visu_buys_sells_in_run(epi,run,run_coin_data,run_data,save_figs=False):
    buy_100=None
    buy_50=None
    sell_100=None
    if save_figs:
        fig_filename=f'fig_e_{epi}_r{run}_buy_sell.png'
        if os.path.exists(os.path.join(save_figs_path,str(epi),fig_filename)):
            return
    fig_4, axs_4 = plt.subplots(figsize=(12, 4))
    axs_4.plot(run_coin_data['date'], run_coin_data['close'])
    min_y = min(run_coin_data['close'])
    max_y = max(run_coin_data['close'])
    for action, action_data in run_data.groupby('action'):
        if action == 'buy_100':
            for row_i, row in action_data.iterrows():
                buy_100 = axs_4.vlines(row['date'], min_y, max_y, color='green')
        if action == 'buy_50':
            for row_i, row in action_data.iterrows():
                buy_50 = axs_4.vlines([row['date']], min_y, max_y, color='orange', label='buy 50 %')
        if action == 'sell':
            for row_i, row in action_data.iterrows():
                sell_100 = axs_4.vlines(row['date'], min_y, max_y, color='red')
    axs_4.legend([buy_100, buy_50, sell_100], ['buy 100%', 'buy 50%', 'sell 100%'])
    fig_4.suptitle(f'Actions in Run {run}')
    if save_figs:
        fig_4.savefig(os.path.join(save_figs_path,str(epi),fig_filename))