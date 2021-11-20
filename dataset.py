import pandas as pd
from pandas import DataFrame
import scipy.io
import os
import re


def get_df(matrix_path, label_path, key1, key2):
    matrix_set = scipy.io.loadmat(matrix_path)[key1]
    label_set = scipy.io.loadmat(label_path)[key2]
    matrix_df = pd.DataFrame(matrix_set)
    label_df = pd.DataFrame(label_set)
    return matrix_df, label_df


def get_dataset(x_train_path, x_test_path, y_train_path, y_test_path):
    x_train_pd, y_train_pd = get_df(x_train_path, y_train_path, 'X_train', 'y_train')
    x_test_pd, y_test_pd = get_df(x_test_path, y_test_path, 'X_test', 'y_test')
    return x_train_pd, y_train_pd, x_test_pd, y_test_pd


if __name__ == '__main__':
    x_train_pd, y_train_pd, x_test_pd, y_test_pd = get_dataset('./Data for Problem 3/X_train_digits.mat',
                                                               './Data for Problem 3/X_test_digits.mat',
                './Data for Problem 3/y_train_digits.mat', './Data for Problem 3/y_test_digits.mat')
    print(x_test_pd.shape, x_train_pd.shape)
    print(y_train_pd.shape, y_test_pd.shape)