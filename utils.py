from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

import numpy as np

def train_test_split(df, labels, tickers):
    data = df[(df['Time'] > (9.5+ 10 / 60) * 60 * 60) & (df['Time'] < (16 - 5/60) * 60 * 60)].reset_index(drop=True)

    train, test = data[data['ticker']!=tickers[-1]].reset_index(drop=True), data[data['ticker']==tickers[-1]].reset_index(drop=True)
    train.drop(['OrderID','ticker'], axis=1, inplace=True)
    test.drop(['OrderID','ticker'], axis=1, inplace=True)

    return train, test   

def train_test_to_csv(train, test, month, day):
    try:
        path_test = './data/'+'train_'+str(month)+'_'+str(day)+ '.csv'
        path_train = './data/'+'test_'+str(month)+'_'+str(day)+ '.csv'
        train.to_csv(path_train, index = False) 
        test.to_csv(path_test, index = False) 
        return 'Well done!'
    except:
        return 'Error'

# train, test  = train_test_split(full_data, labels, tickers[:-1])


def get_label(df, tk):
    y = df.iloc[:, -tk:]
    return np.array(y)

def get_feature(df, feature_final):
    x = df.loc[:, feature_final]
    x = normalize(x)
    return np.array(x)

def get_classfication(x, y, T):
    [n, d] = x.shape
    dy = y[T-1:n]
    dx = np.ones((n-T+1, T, d))
    for i in range(T, n+1):
        dx[i-T] = x[i-T:i, :]
    return dx.reshape(dx.shape+(1,)), dy


def generate_x_y(df, tk, k, T, feature_final):
    x = get_feature(df, feature_final)
    y = get_label(df, k)
    x, y = get_classfication(x, y, T)
    y = y[:, k-1]
    y = np_utils.to_categorical(y, 3)
    return x, y


def train_val_split(data, perc=0.8):
    num = int(data.shape[0]*0.8)
    train = data[:num]
    val = data[num:]
    return train, val
