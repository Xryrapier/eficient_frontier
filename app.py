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
model_path_name = os.path.expanduser('~')+'/code/Xryrapier/eficient_frontier/finalized_model.sav'
app.state.model = load(open(model_path_name, 'rb'))

# $WIPE_END


@app.get("/predict")
def predict(

):
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # $CHA_BEGIN

    X_pred=pd.DataFrame([[2, 32, 4, 1, 0, 5, 4, 2000, 6, 0, 8000]] ,\
                columns=['HHSEX', 'AGE', 'EDCL', 'MARRIED', 'KIDS', 'FAMSTRUCT', 'OCCAT1','INCOME', 'WSAVED', 'YESFINRISK', 'NETWORTH'])

    sigma = pred(X_pred)
    fin_pd, res = get_actions_opt_portfolio(ndays=20, invest=100000, sigma = sigma)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON

    return {'fin_pd': fin_pd,
            'sigma': list(sigma)}


@app.get("/")
def root():
    # $CHA_BEGIN
    return {'ok': True}
    # $CHA_END
