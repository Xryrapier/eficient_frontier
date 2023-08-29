import numpy as np
from typing import Tuple



def initialize_model(input_shape: tuple):
    """
    Initialize the MODEL
    """
    pass # your code


    print("✅ Model initialized")

    return "model"


def train_model(
        model,
        X: np.ndarray,
        y: np.ndarray,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split= validation_split
    )

    print(f"✅ Model trained on {len(X)} rows")

    return model, history
