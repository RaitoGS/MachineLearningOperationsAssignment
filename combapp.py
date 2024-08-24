import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import requests
import pickle
import pycaret
from pycaret.anomaly import AnomalyExperiment
import numpy as np
import math
from pycaret.regression import load_model as load_model_reg, predict_model as predict_model_reg
from pycaret.classification import load_model as load_model_class, predict_model as predict_model_class

def main():
    st.set_page_config(page_title="Mushroom Classification Prediction", layout="wide")
    st.title("MLOps Prediction Appliation")
    tab1, tab2, tab3 = st.tabs(["Regression Problem", "Classification Problem", "Anomaly Problem"])

    with tab3:
        st.title("Financial Statement Anomaly Detection")
        with st.form("financial_form"):
            st.title("Enter financial statement below")
            fiscal_yr = st.selectbox("Fiscal Year", [2017, 2018, 2019, 2020, 2021, 2022])
            fiscal_mth = st.selectbox("Fiscal Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            dept_name = st.text_input("Department Name")
            div_name = st.selectbox("Division Name",['department of education', 'education block grants',
            'special needs programs', 'advisory council', 'transportation',
            'driver training', 'other items', 'e education block grants',
            'e transportation', 'academic support', 'operations support',
            'office of early learning', 'office of the secretary',
            'student support', 'professional standards board',
            'state board of education', 'public school transportation',
            'educator support', 'adult education and work force'])
            merchant = st.text_input("Merchant")
            cat_desc = st.text_input("Category Description")
            trans_dt = st.date_input("Transaction Date")
            amt = st.number_input("Amount")

            submitted = st.form_submit_button("Submit")

        if submitted:
            data = {
                "FISCAL_YR": [fiscal_yr],
                "FISCAL_MTH": [fiscal_mth],
                "DEPT_NAME": [dept_name],
                "DIV_NAME": [div_name],
                "MERCHANT": [merchant],
                "CAT_DESC": [cat_desc],
                "TRANS_DT": [trans_dt],
                "AMT": [amt]
            }

            df = pd.DataFrame(data)
            df.DIV_NAME = df.DIV_NAME.str.lower()
            df.MERCHANT = df.MERCHANT.str.lower()
            df.CAT_DESC = df.CAT_DESC.str.lower()
            df['TRANS_DT'] = pd.to_datetime(df['TRANS_DT'],errors='coerce')
            df['TRANS_DAY'] = df['TRANS_DT'].dt.dayofweek
            df['TRANS_MONTH'] = df['TRANS_DT'].dt.month
            df['TRANS_QUARTER'] = df['TRANS_DT'].dt.quarter
            df['TRANS_YEAR'] = df['TRANS_DT'].dt.year
            df.dropna(subset=['TRANS_DT'], inplace=True)
            st.write("Form submitted successfully!")
            st.write(df)

            exp = AnomalyExperiment()
            model = exp.load_model('local_anomaly_detect')

            model_pred = exp.predict_model(model, data=df)
            if model_pred.Anomaly[0] == 1:
                st.write("Alert! Anomaly Detected")
            else:
                st.write("No Anomaly Detected")

    with tab2:
        st.title("Mushroom Classification Prediction")
        model = load_model_class(r'irsyad\artifacts\mushroom_classification_model')

        # Create three columns
        col1, col2, col3 = st.columns(3)

        # First column dropdowns
        with col1:
            cap_shape = st.selectbox('Cap Shape', ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'])
            cap_surface = st.selectbox('Cap Surface', ['fibrous', 'grooves', 'scaly', 'smooth'])
            cap_color = st.selectbox('Cap Color', ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'])
            bruises = st.selectbox('Bruises', ['bruises', 'no'])
            odor = st.selectbox('Odor', ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'])
            population = st.selectbox('Population', ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'])
            habitat = st.selectbox('Habitat', ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'])

        # Second column dropdowns
        with col2:
            gill_attachment = st.selectbox('Gill Attachment', ['attached', 'free'])
            gill_spacing = st.selectbox('Gill Spacing', ['close', 'crowded'])
            gill_size = st.selectbox('Gill Size', ['broad', 'narrow'])
            gill_color = st.selectbox('Gill Color', ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])
            stalk_shape = st.selectbox('Stalk Shape', ['enlarging', 'tapering'])
            stalk_root = st.selectbox('Stalk Root', ['bulbous', 'club', 'equal', 'rooted', 'missing'])
            spore_print_color = st.selectbox('Spore Print Color', ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'])

        # Third column dropdowns
        with col3:
            stalk_surface_above_ring = st.selectbox('Stalk Surface Above Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
            stalk_surface_below_ring = st.selectbox('Stalk Surface Below Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
            stalk_color_above_ring = st.selectbox('Stalk Color Above Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
            stalk_color_below_ring = st.selectbox('Stalk Color Below Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
            veil_type = st.selectbox('Veil Type', ['partial', 'universal'])
            veil_color = st.selectbox('Veil Color', ['brown', 'orange', 'white', 'yellow'])
            ring_number = st.selectbox('Ring Number', ['none', 'one', 'two'])
            ring_type = st.selectbox('Ring Type', ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'])

        # When the user clicks the button
        if st.button('Predict'):
            # Prepare the data as a dictionary
            input_data = {
                'cap_shape': cap_shape,
                'cap_surface': cap_surface,
                'cap_color': cap_color,
                'bruises': bruises,
                'odor': odor,
                'gill_attachment': gill_attachment,
                'gill_spacing': gill_spacing,
                'gill_size': gill_size,
                'gill_color': gill_color,
                'stalk_shape': stalk_shape,
                'stalk_root': stalk_root,
                'stalk_surface_above_ring': stalk_surface_above_ring,
                'stalk_surface_below_ring': stalk_surface_below_ring,
                'stalk_color_above_ring': stalk_color_above_ring,
                'stalk_color_below_ring': stalk_color_below_ring,
                'veil_type': veil_type,
                'veil_color': veil_color,
                'ring_number': ring_number,
                'ring_type': ring_type,
                'spore_print_color': spore_print_color,
                'population': population,
                'habitat': habitat
            }

            input_df = pd.DataFrame([input_data])
            # Ensure the input_df has only the feature columns (excluding 'class')
            feature_columns = [col for col in model.feature_names_in_ if col != 'class']
            input_df = input_df.reindex(columns=feature_columns)

            # Generate predictions
            predictions = predict_model_class(model, data=input_df)
            
            # Extract the prediction result
            prediction = predictions['prediction_label'][0]
            score = predictions['prediction_score'][0]
            st.write(f"Prediction: {prediction}")
            st.write(f"Confidence Score: {score}")
    
    with tab1:
        model = load_model_reg('kavan\\Mlops\\singapore_hdb_resale_price_pipeline_final')
        # Function to predict the resale price
        def predict(data):
            predictions = predict_model_reg(model, data=data)
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

main()