import numpy as np
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.compose import make_column_selector


def preprocess_features() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset
        into a preprocessed one.
        Stateless operation: "fit_transform()" equals "transform()".
        """

        # Numerical features
        robust_features = ['INCOME', 'NETWORTH']
        standard_features = ['AGE']
        minmax_features = ['KIDS']

        scalers = ColumnTransformer([
            ("robust_scaler", RobustScaler(), robust_features),
            ("standard_scaler", StandardScaler(), standard_features),
            ("minmax_scaler", MinMaxScaler(), minmax_features),
        ], remainder="passthrough").set_output(transform='pandas')

        num_transformer = Pipeline([
            ("num_imputer", SimpleImputer(strategy="median")),
            ("scalers", scalers)
        ])

        # Categorical features
        ordinal_features = []
        ohe_features = ['HHSEX', 'MARRIED', 'OCCAT1', 'WSAVED', 'FAMSTRUCT']

        encoders = ColumnTransformer([
            ("ordinalencoder", OrdinalEncoder(), ordinal_features),
            ("onehotencoder", OneHotEncoder(drop='if_binary',
                                            sparse_output=False,
                                            handle_unknown="ignore",
                                            min_frequency=5), ohe_features),
        ], remainder="passthrough")

        cat_transformer = Pipeline([
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("encoders", encoders)
        ])

        # Column Transformer
        num_features = ['AGE', 'KIDS', 'INCOME', 'NETWORTH']
        cat_features = ['HHSEX', 'MARRIED', 'OCCAT1', 'WSAVED', 'FAMSTRUCT', 'YESFINRISK', 'EDCL']

        final_preprocessor = ColumnTransformer([
            ("num_transformer", num_transformer, num_features),
            ("cat_transformer", cat_transformer, cat_features),
        ], remainder="passthrough").set_output(transform="pandas")

        return final_preprocessor


def preprocess(data_normal: pd.DataFrame) -> np.ndarray:

    print("\nPreprocessing features...")
    # Train test split
    y = data_normal['RT']
    X = data_normal.drop(columns=['RT'])
    preprocessor = preprocess_features()
    preprocessor.fit(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    print("✅ X_train_preprocessed, with shape", X_train_preprocessed.shape)
    print("✅ X_test_preprocessed, with shape", X_test_preprocessed.shape)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor
