import copy
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def calculate_risk_tolerance(dataset):
    dataset['RiskFree'] = dataset['LIQ'] + dataset['CDS'] + dataset['SAVBND'] + dataset['CASHLI']
    dataset['Risky'] = dataset['NMMF'] + dataset['STOCKS'] + dataset['BOND']
    dataset['RT'] = dataset['Risky'] / (dataset['Risky'] + dataset['RiskFree'])
    return dataset

def copy_and_show_head(dataset):
    dataset2 = copy.deepcopy(dataset)       
    return dataset2

def clean_dataset(dataset):
    dataset_cleaned = dataset.dropna(axis=0)
    dataset_cleaned = dataset_cleaned[~dataset_cleaned.isin([np.nan, np.inf, -np.inf]).any(1)]
    return dataset_cleaned

def check_null_values(dataset):
    return dataset.isnull().values.any()

def select_and_drop_columns(dataset, keep_list):
    drop_list = [col for col in dataset.columns if col not in keep_list]
    dataset_cleaned = dataset.drop(labels=drop_list, axis=1, inplace=False)
    return dataset_cleaned

def plot_scatter_matrix(dataset, figsize=(15, 15)):
    scatter_matrix(dataset, figsize=figsize)
    plt.show()

def split_dataset(dataset, target_column, test_size, random_state):
    Y = dataset[target_column]
    X = dataset.loc[:, dataset.columns != target_column]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_validation, Y_train, Y_validation
