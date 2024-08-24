import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

import pickle
import pycaret
from pycaret.anomaly import AnomalyExperiment

st.set_page_config(page_title="TesterMk1", layout="wide")
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
