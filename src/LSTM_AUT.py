import os
import numpy as np
import pandas as pd
from tensorflow import keras
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
import yfinance as yf
import pathlib

REPO_BASE_DIR =  pathlib.Path(__file__).parent.parent.absolute().as_posix() # ./ai_trader directory


def to_sequences(data, seq_len, forecast):
    """
    Builds all sequences of data with a length of seq_len

    NOTES
    -----
    forecast is not used

    """
    d = []
    lst = list(range(len(data) - seq_len))
    for index in lst:
        d.append(data[index: index + seq_len])
    return np.array(d)


def preprocess_sequence(data_raw, seq_len, train_sequence, test_sequence, forecast):
    """
    preprocesses sequences to train and test LSTM

    NOTES
    -----
    X_train:
    Builds array of (train_sequence) sequences, with a length of (seq_len - forecast)
    """
    data = to_sequences(data_raw, seq_len, forecast) # 700x300
    test_end = data.shape[0] # 700
    test_start = test_end - (test_sequence) # 600
    train_end = test_start - 1 # 599
    train_start = train_end - (train_sequence) # 549

    X_train = data[train_start:train_end, :SEQ_LEN-forecast, :]
    y_train = data[train_start:train_end, SEQ_LEN-forecast:, :]

    X_test = data[test_start:test_end, :SEQ_LEN-forecast, :]
    y_test = data[test_start:test_end, SEQ_LEN-forecast:, :]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Choose stock with ticker symbol
    #shortcut = 'AAPL'
    #dir_path = shortcut
    #df = yf.download(shortcut)
    btc_data_path = os.path.join(REPO_BASE_DIR, 'data/BTC_histData_dt1800.0s_20220825_0629.csv')
    data = pd.read_csv(btc_data_path)
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    #print(df.head())

    # Normalization
    scaler = MinMaxScaler()
    close_price = df.close.values.reshape(-1, 1)
    scaled_close = scaler.fit_transform(close_price)

    # Handle NaNs
    scaled_close = scaled_close[~np.isnan(scaled_close)]
    scaled_close = scaled_close.reshape(-1, 1)

    # Build sequences
    SEQ_LEN = 300
    forecast = int(0.33*SEQ_LEN)

    test_sequence = forecast
    train_sequence_list = [50,100,200,400] #,1080]

    for train_sequence in train_sequence_list:
        print('[INFO]: Train sequence: ' + str(train_sequence))

        X_train, y_train, X_test, y_test = preprocess_sequence(scaled_close, SEQ_LEN, train_sequence, test_sequence, forecast)

        #Create LSTM
        DROPOUT = 0.2
        WINDOW_SIZE = SEQ_LEN - forecast
        
        model = keras.Sequential()

        model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True), input_shape=(WINDOW_SIZE, X_train.shape[-1])))
        model.add(Dropout(rate=DROPOUT))
        model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
        model.add(Dropout(rate=DROPOUT))
        model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
        model.add(Dense(units=1))
        model.add(Activation('linear'))

        #Train model
        BATCH_SIZE = 64

        model.compile(loss='mean_squared_error', optimizer='adam')

        history = model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_split=0.1
        )

        # Initiliaze plot
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)

        y_hat = model.predict(X_test)

        X_test_inverse = scaler.inverse_transform(X_test[test_sequence-1,:,:])
        y_hat_inverse = scaler.inverse_transform(y_hat)
        y_test_inverse = scaler.inverse_transform(y_test[test_sequence-1,:,:])

        #Predict future without labels
        y_dark = model.predict(y_test)
        y_dark_inverse = scaler.inverse_transform(y_dark)

        #Create Array offset array
        z_test = np.full(y_hat_inverse.shape, y_hat_inverse[0,0])

        #Concat
        X_test_y_test_inverse = np.vstack((X_test_inverse, y_test_inverse))
        y_hat_dark_inverse = np.vstack((z_test, y_hat_inverse, y_dark_inverse))

        plt.plot(y_test_inverse, label="Real", color='green')
        plt.plot(y_hat_inverse, label="Prediction", color='red')
        plt.title('Predicted Price')
        plt.xlabel('Time [30 min]')
        #plt.xticks(np.arange(forecast)[::2], np.arange(1:(forecast//2 + 1))) # 30 min data 
        plt.xticks(np.arange(forecast)[::2], list(np.arange(1, 51, 1)))
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.show();
        plt.close()

        y_test_inverse_norm = y_test_inverse/y_test_inverse[0] - 1
        y_hat_inverse_norm = y_hat_inverse/y_hat_inverse[0] - 1
        y_dark_inverse_norm = y_dark_inverse / y_hat_inverse[0] - 1

        y_hat_dark_inverse_norm = np.append(y_hat_inverse_norm, y_dark_inverse_norm)

        dir_path = os.path.join(REPO_BASE_DIR, 'data/LSTM/BTC')
        plt.plot(y_test_inverse_norm, label="Real", color='green')
        plt.plot(y_hat_dark_inverse_norm, label="Prediction", color='red')
        plt.title('Predicted Relative Price')
        plt.xlabel('Time [min]')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.savefig(dir_path + '_' + str(train_sequence) + '.png')
        plt.close()