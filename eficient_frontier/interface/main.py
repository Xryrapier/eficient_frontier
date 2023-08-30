import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.parser import parse
from eficient_frontier.ml_logic.data import *
from eficient_frontier.ml_logic.preprocessor import *
from eficient_frontier.ml_logic.model import *
from eficient_frontier.ml_logic.sp500_data import get_sp500_data
from eficient_frontier.params import *

# def preprocess_and_train():
#     """
#     - Clean and preprocess data
#     - Train a Keras model on it
#     - Save the model
#     - Compute & save a validation performance metric
#     """

#     print("\n ⭐️ preprocess_and_train")


#     # Clean data using data.py
#     data=clean_data(data)

#     # Create (X_train, y_train, X_val, y_val) without data leaks


#     # Create (X_train_processed, X_val_processed) using `preprocessor.py`
#     X_train_processed= preprocess_features("X_train")
#     X_val_processed=preprocess_features("X_val")

#     # Train a model on the training set, using `model.py`
#     model = None
#     model = initialize_model(input_shape=X_train_processed.shape[1:])
#     model , history = train_model(model,X_train_processed,"y_train" , validation_data=(X_val_processed,"y_val"))

#     # Save trained model
#     pass #your code here

#     print("✅ preprocess_and_train() done")


# def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
#     print("\n ⭐️  pred" )

#     if X_pred is None:
#         X_pred = pd.DataFrame(dict(
#             # defualt data to predict
#         ))

#     model = load_model()
#     X_processed = preprocess_features(X_pred)
#     y_pred = model.predict(X_processed)

#     print(f"✅ pred() done")

#     return y_pred


if __name__ == '__main__':
    try:
        tickers, sp500_data = get_sp500_data()
        get_best_5_stock(tickers, sp500_data)
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
