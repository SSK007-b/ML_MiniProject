import streamlit as st
import numpy as np
import joblib

# Load the models and other utilities
scaler = joblib.load('pklFiles\scaler.pkl')
encoder = joblib.load('pklFiles\Encode.pkl')
model = joblib.load('pklFiles\Qpistachio_ensemble.pkl')

# Define the features required
FEATURES = [
    "AREA", "PERIMETER", "MINOR_AXIS", "ECCENTRICITY", 
    "EQDIASQ", "SOLIDITY", "EXTENT", "ROUNDNESS", 
    "SHAPEFACTOR_1", "SHAPEFACTOR_2"
]

# Streamlit UI
st.title("Pistachio Classification App")
st.write("Enter the values for the following features to classify the type of pistachio.")

# Collect user input for each feature
user_input = []
for feature in FEATURES:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_input.append(value)

# Prediction button
if st.button("Predict"):
    # Convert user input to a 2D numpy array
    input_array = np.array([user_input])

    # Transform input using the scaler
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Decode the predicted label
    predicted_label = encoder.inverse_transform(prediction)

    # Display the result
    st.success(f"The predicted pistachio type is: {predicted_label[0]}")