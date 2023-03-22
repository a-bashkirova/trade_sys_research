import pandas as pd
import numpy as np
from src.constants import WINDOWS
import torch
from torch.utils.data import DataLoader, TensorDataset


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


def prepare_dataloader(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_data = train_data.to(device)
    train_target = torch.tensor(y_train)
    train_dataloader = DataLoader(TensorDataset(train_data, train_target), batch_size=len(train_data))
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_data = test_data.to(device)
    test_target = torch.tensor(y_test)
    test_dataloader = DataLoader(TensorDataset(test_data, test_target), batch_size=len(test_data))
    return train_dataloader, test_dataloader

