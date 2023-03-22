import pandas as pd
import numpy as np
from src.constants import WINDOWS


def prepare_data(train_data: pd.DataFrame, company : str, mode='regression'):
    X_train, y_train = np.array([]).reshape(-1, WINDOWS[company]), np.array([]).reshape(-1, 1)
    n = train_data.close.values.shape[0]
    for i in range(n - WINDOWS[company] - 1):
        X_train = np.append(X_train, np.array(train_data.close.values[i:i + WINDOWS[company]], ndmin=2), axis=0)
        if mode == 'regression':
            y_train = np.append(y_train, np.array([train_data.close.values[i + WINDOWS[company]]], ndmin=2), axis=0)
        else:
            flg = int(train_data.close.values[i + WINDOWS[company] - 1] < train_data.close.values[i + WINDOWS[company]])
            y_train = np.append(y_train, np.array([flg], ndmin=2), axis=0)
    return X_train, y_train


def date2data(company : str, req_date : str):
    temp_data = pd.read_csv(f'data/{company}.csv')
    temp_data.drop(labels=['<PER>', '<TIME>'], axis=1, inplace=True)
    temp_data.rename({'<TICKER>': 'company', '<CLOSE>': 'close', '<DATE>': 'date'}, axis=1, inplace=True)
    temp_data['date'] = pd.to_datetime(temp_data['date'], dayfirst=True)
    temp_data = temp_data.loc[temp_data['date'] <= pd.to_datetime(req_date)]
    temp_data = temp_data.tail(WINDOWS[company])
    return temp_data.close.values.reshape(1, -1)


def available_dates(company : str):
    temp_data = pd.read_csv(f'data/{company}.csv')
    temp_data.drop(labels=['<PER>', '<TIME>'], axis=1, inplace=True)
    temp_data.rename({'<TICKER>': 'company', '<CLOSE>': 'close', '<DATE>': 'date'}, axis=1, inplace=True)
    temp_data['date'] = pd.to_datetime(temp_data['date'], dayfirst=True)
    return min(temp_data['date']).date(), max(temp_data['date']).date()
