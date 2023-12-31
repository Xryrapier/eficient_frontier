from eficient_frontier.ml_logic.data import *
from eficient_frontier.ml_logic.preprocessor import *
from eficient_frontier.ml_logic.finfuncs import *


import pandas as pd
# $WIPE_BEGIN

from eficient_frontier.interface.main import preproces_train_evaluate_save_model
from eficient_frontier.ml_logic.preprocessor import preprocess_features
from eficient_frontier.interface.main import pred
# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
model_path_name = 'models/finalized_model.sav'
# model_path_name = os.path.expanduser('~')+'/code/Xryrapier/eficient_frontier/finalized_model.sav'

app.state.model = load(open(model_path_name, 'rb'))

# $WIPE_END


@app.get("/predict")
def predict(
    HHSEX: int,
    AGE: int,
    EDCL: int,
    MARRIED: int,
    KIDS: int,
    FAMSTRUCT: int,
    OCCAT1: int,
    INCOME: float,
    WSAVED: int,
    YESFINRISK: int,
    NETWORTH: float,
    NDAYS: int,
    AMOUNT: int
):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # $CHA_BEGIN


    user_params = pd.DataFrame(locals(), index=[0])
    X_pred_api = user_params.loc[: ,"HHSEX" : "NETWORTH"]
    sigma = pred(X_pred_api)
    fin_pd, res = get_actions_opt_portfolio(ndays=user_params["NDAYS"][0], invest=user_params["AMOUNT"][0], sigma = sigma)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON

    return {'res': res,
            'sigma': list(sigma)}


@app.get("/")
def root():
    # $CHA_BEGIN
    return {'ok': True}
    # $CHA_END
