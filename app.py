import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and feature names
model = joblib.load('logreg_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')  # Ensure this file contains the correct feature names

# Function to preprocess user input
def preprocess_input(data):
    # Perform one-hot encoding for categorical features
    data = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel'], drop_first=True)

    # Map class categories to integers
    class_mapping = {"Eco": 0, "Eco Plus": 1, "Business": 2}
    data['Class'] = data['Class'].map(class_mapping)

    # Add missing features with default values
    for feature in feature_names:
        if feature not in data.columns:
            data[feature] = 0

    # Reorder the columns to match the training order
    data = data[feature_names]

    return data

# Function to make predictions
def make_prediction(input_data):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Scale the input data
    scaled_data = scaler.transform(processed_data)

    # Predict using the loaded model
    prediction = model.predict(scaled_data)
    return prediction

# Streamlit app
def main():
    st.title("Airline Passenger Satisfaction Prediction")

    # Collect user input
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    flight_distance = st.number_input("Flight Distance", min_value=1, max_value=10000, value=1000)
    departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
    arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    customer_type = st.selectbox("Customer Type", options=["Loyal Customer", "Disloyal Customer"])
    travel_type = st.selectbox("Type of Travel", options=["Business", "Personal"])
    travel_class = st.selectbox("Class", options=["Eco", "Eco Plus", "Business"])

    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'Age': [age],
        'Flight Distance': [flight_distance],
        'Departure Delay in Minutes': [departure_delay],
        'Arrival Delay in Minutes': [arrival_delay],
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Type of Travel': [travel_type],
        'Class': [travel_class]
    })

    # Display the original user input data
    st.write("User Input Data:", input_data)

    # Button to make prediction
    if st.button("Predict Satisfaction"):
        # Preprocess and encode the user input
        processed_data = preprocess_input(input_data)

        # Display the preprocessed data
        # st.write("Preprocessed Data:", processed_data)

        # Make the prediction
        prediction = make_prediction(input_data)

        # Display the prediction result
        if prediction[0] == 1:
            st.success("The passenger is satisfied.")
        else:
            st.error("The passenger is not satisfied.")

if __name__ == "__main__":
    main()
