import streamlit as st
from  params import *
import pandas as pd
import numpy as np
import os
from ml_logic.data import *
from ml_logic.preprocessor import *
from ml_logic.model import *
from ml_logic.finfuncs import *
from params import *



investors = pd.read_csv(ROOT_DIR+'/raw_data/InputData.csv', index_col = 0 )
assets = pd.read_csv(ROOT_DIR+'/raw_data/SP500Data.csv',index_col=0)
missing_fractions = assets.isnull().mean().sort_values(ascending=False)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
assets.drop(labels=drop_list, axis=1, inplace=True)
# Fill the missing values with the last value available in the dataset.
assets=assets.fillna(method='ffill')
options=np.array(assets.columns)
# str(options)
options = []
for tic in assets.columns:
    #{'label': 'user sees', 'value': 'script sees'}
    mydict = {}
    mydict['label'] = tic #Apple Co. AAPL
    mydict['value'] = tic
    options.append(mydict['label'])



# Define the layout
st.title('Robo Advisor Dashboard')

st.markdown('## Step 1 : Enter Investor Characteristics')

age = st.slider('Age:', min_value=investors['AGE07'].min(), max_value=70, value=25)
net_worth = st.slider('NetWorth:', min_value=-1000000, max_value=3000000, value=10000)
income = st.slider('Income:', min_value=-1000000, max_value=3000000, value=100000)
education = st.slider('Education Level (scale of 4):', min_value=1, max_value=4, value=2)
married = st.slider('Married:', min_value=1, max_value=2, value=1)
kids = st.select_slider('Kids:', options=investors['KIDS07'].unique(), value=3)
occupation = st.slider('Occupation:', min_value=1, max_value=4, value=3)
risk_tolerance = st.slider('Willingness to take Risk:', min_value=1, max_value=4, value=3)

calculate_button = st.button('Calculate Risk Tolerance', key='investor_char_button')

st.markdown('## Step 2 : Asset Allocation and Portfolio Performance')

risk_tolerance_text = st.text_input('Risk Tolerance (scale of 100) :')
# ticker_symbols = st.multiselect('Select the assets for the portfolio:', options=options, default=['GOOGL', 'FB', 'GS', 'MS', 'GE', 'MSFT'])
ticker_symbols = st.multiselect('Select the assets for the portfolio:', options=options)

submit_button = st.button('Submit', key='submit-asset_alloc_button')

# st.subheader('Asset Allocation')
# st.plotly_chart(asset_allocation_fig)

# st.subheader('Performance')
# st.plotly_chart(performance_fig)
