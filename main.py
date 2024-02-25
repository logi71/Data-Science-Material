import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle
st.header("Patient care classifier system")
with open('random_forest_model.pkl','rb')as model_file:
    loaded_model=pickle.load(model_file)

st_haematocrit= st.number_input("Enter HAEMATOCRIT value: ")
st_HAEMOGLOBINS= st.number_input("Enter HAEMOGLOBINS value: ")
st_ERYTHROCYTE= st.number_input("Enter ERYTHROCYTE value: ")
st_LEUCOCYTE= st.number_input("Enter LEUCOCYTE value: ")
st_THROMBOCYTE= st.number_input("Enter THROMBOCYTE value: ")
st_MCH= st.number_input("Enter MCH value: ")
st_MCHC= st.number_input("Enter MCHC value: ")
st_MCV= st.number_input("Enter MCV value: ")
st_AGE= st.number_input("Enter AGE value: ")
st_SEX= st.number_input("Enter SEX value (Enter 1 for male and 0 for female ): ")

user_data=[[st_haematocrit,st_HAEMOGLOBINS,st_ERYTHROCYTE,st_LEUCOCYTE,st_THROMBOCYTE,
st_MCH,st_MCHC,st_MCV,st_AGE,st_SEX]]

cols = [
    [
        "HAEMATOCRIT",
        "HAEMOGLOBINS",
        "ERYTHROCYTE",
        "LEUCOCYTE",
        "THROMBOCYTE",
        "MCH",
        "MCHC",
        "MCV",
        "AGE",
        "SEX",
    ]
]

pd_test_df = pd.DataFrame(user_data, columns=cols)

#st.write(pd_test_df)
# Display user input
st.subheader("User Input")
st.write(pd_test_df)

# Make predictions using the loaded Random Forest model
rf_predict_user_data = loaded_model.predict(user_data)

st.write(rf_predict_user_data)