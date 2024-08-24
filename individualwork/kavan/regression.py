import streamlit as st
import numpy as np
import pandas as pd
import math
from pycaret.regression import load_model, predict_model

# Load the trained model
model = load_model('C:\\Users\\gideo\\SchoolStuff\\DAAA Y3\\MLO\\Assignment\\kavan\\Mlops\\singapore_hdb_resale_price_pipeline_final')

# Function to predict the resale price
def predict(data):
    predictions = predict_model(model, data=data)
    print(predictions.columns)  # This will print the column names to the console
    logged_prediction = predictions['prediction_label'].iloc[0]  # Corrected column name
    actual_prediction = math.exp(logged_prediction)
    return actual_prediction

# Calculate age of the flat
def calculate_age(lease_commence_date):
    current_year = pd.Timestamp.now().year
    return current_year - lease_commence_date

# Streamlit application layout and input handling
st.title('HDB Resale Flat Price Prediction')
st.write('Enter the details of the HDB flat to predict the resale price.')

block = st.text_input('Block')
street_name = st.text_input('Street Name')
town = st.text_input('Town')
postal_code = st.text_input('Postal Code')
month = st.selectbox('Month', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
flat_type = st.selectbox('Flat Type', ['1 Room', '2 Room', '3 Room', '4 Room', '5 Room', 'Executive', 'Multi-Generation'])
storey_range = st.selectbox('Storey Range', ['01 to 03', '04 to 06', '07 to 09', '10 to 12', '13 to 15', '16 to 18', '19 to 21', '22 to 24', '25 to 27', '28 to 30', '31 to 33', '34 to 36', '37 to 39', '40 to 42', '43 and above'])
floor_area = st.number_input('Floor Area (sqm)', min_value=0)
flat_model = st.text_input('Flat Model')
lease_commence_date = st.number_input('Lease Commence Date', min_value=1960, max_value=2020, step=1)
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')
cbd_dist = st.number_input('Distance to CBD (km)', min_value=0.0, max_value=20000.0, step=0.1, format="%.2f")
min_dist_mrt = st.number_input('Distance to Nearest MRT (km)', min_value=0.0, max_value=10000.0, step=0.1, format="%.2f")
age = calculate_age(lease_commence_date)

if st.button('Predict Resale Price'):
    input_features = pd.DataFrame([[
        block, street_name, town, postal_code, month, flat_type, storey_range, floor_area,
        flat_model, lease_commence_date, latitude, longitude, cbd_dist, min_dist_mrt, age
    ]], columns=[
        'block', 'street_name', 'town', 'postal_code', 'month', 'flat_type', 'storey_range',
        'floor_area_sqm', 'flat_model', 'lease_commence_date', 'latitude', 'longitude',
        'cbd_dist', 'min_dist_mrt', 'age'
    ])

    prediction = predict(input_features)
    st.success(f'The predicted resale price is ${prediction:,.2f}')
