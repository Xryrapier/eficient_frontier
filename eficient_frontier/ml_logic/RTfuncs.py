import copy
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
from pickle import dump, load

def calculate_risk_tolerance(dataset):
    dataset['RiskFree'] = dataset['LIQ'] + dataset['CDS'] + dataset['SAVBND'] + dataset['CASHLI']
    dataset['Risky'] = dataset['NMMF'] + dataset['STOCKS'] + dataset['BOND']
    dataset['RT'] = dataset['Risky'] / (dataset['Risky'] + dataset['RiskFree'])
    return dataset

def deep_copy(dataframe):
    copied_dataframe = copy.deepcopy(dataframe)

def preprocess_dataset(dataset):
    # Drop rows with any NaN values
    dataset = dataset.dropna(axis=0)
    # Remove rows containing infinite values
    dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    # Check for any remaining null values
    null_values_exist = dataset.isnull().values.any()
    return dataset, null_values_exist

def preprocess_and_drop_columns(dataset, keep_list):
    df = dataset.copy()
    drop_list = [col for col in df.columns if col not in keep_list]
    df.drop(labels=drop_list, axis=1, inplace=True)
    return df

def prepare_train_validation_sets(dataframe, target_column, test_size=0.2, random_state=None):
    Y = dataframe[target_column]
    X = dataframe.drop(columns=[target_column])
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_validation, Y_train, Y_validation

def create_regression_models():
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('EN', ElasticNet()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('CART', DecisionTreeRegressor()))
    models.append(('SVR', SVR()))
    models.append(('ABR', AdaBoostRegressor()))
    models.append(('GBR', GradientBoostingRegressor()))
    models.append(('RFR', RandomForestRegressor()))
    models.append(('ETR', ExtraTreesRegressor()))
    return models

def perform_cross_validation_and_store_results_with_best_model(models, X_train, Y_train, num_folds=10, seed=None, scoring='neg_mean_squared_error'):
    results = []
    names = []
    best_score = float('-inf')
    best_model = None

    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        cv_results = -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        mean_score = cv_results.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return names, results, best_model

def perform_grid_search_random_forest(X_train, Y_train, best_model, num_folds=10, seed=None, scoring='neg_mean_squared_error'):
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400]}
    model = best_model(random_state=seed)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def create_and_fit_model(X_train, Y_train, model_class, **model_params):
    model = model_class(**model_params)
    model.fit(X_train, Y_train)
    return model

def calculate_r2_score(predictions, true_values):
    r2 = r2_score(true_values, predictions)
    return r2

def evaluate_regression_model(model, X_validation, Y_validation):
    predictions = model.predict(X_validation)
    mse = mean_squared_error(Y_validation, predictions)
    r2 = r2_score(Y_validation, predictions)
    return mse, r2

def save_model_to_pickle(model, filename):
    with open(filename, 'wb') as model_file:
        dump(model, model_file)
    print("Model saved as", filename)

from pickle import load
from sklearn.metrics import mean_squared_error, r2_score

def load_and_evaluate_model(model_filename, X_validation, Y_validation):
    # Load the model from the pickle file
    loaded_model = load(open(model_filename, 'rb'))

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X_validation)

    # Calculate and print performance metrics
    mse = mean_squared_error(Y_validation, predictions)
    r2 = r2_score(Y_validation, predictions)

    #print("R^2 Score:", r2)
    #print("Mean Squared Error:", mse)
