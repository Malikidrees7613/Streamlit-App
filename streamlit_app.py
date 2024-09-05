import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

model= joblib.load('K-Nearest Neighbors.pkl')

with open('accuracy.txt','r') as file:
    accuracy= file.read()
st.title("Model Accuracy and Real-Time Prediction")
st.write(f"Model {accuracy}")

st.header("Real-Time Prediction")

test_data = pd.read_csv('mobile_price_range_data.csv')

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

input_data = []
for col in X_test.columns:
    input_value = st.number_input(f"Input for {col}", value=0.0)
    input_data.append(input_value)

input_df = pd.DataFrame([input_data], columns=X_test.columns)

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction[0]}")

st.header("Accuracy Plot")
st.bar_chart([float(accuracy.split(': ')[1])])