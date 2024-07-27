# Airline Passenger Satisfaction Prediction

This project involves predicting airline passenger satisfaction using machine learning. The goal is to determine whether a passenger is satisfied or not based on various features such as age, flight distance, delay times, and more.

## Project Overview

The dataset used for this project contains information about airline passengers and their satisfaction levels. A logistic regression model has been trained to predict whether a passenger will be satisfied or not based on the provided features.

## Files

- `data/`: Contains the dataset used for training and prediction.
- `model/`: Contains the trained model and preprocessing objects.
- `app.py`: The Streamlit app for user interface and prediction.
- `requirements.txt`: List of Python dependencies required for the project.
- `README.md`: This file.

## Model Evaluation

The logistic regression model achieved the following performance metrics:

- **Accuracy**: 87.6% on the training set, 88% on the test set
- **F1-Score**: 0.85
- **Recall**: 0.85
- **ROC AUC**: 0.93

## Usage

1. **Open the Streamlit app** in your browser.
2. **Input the required information** such as age, flight distance, delay times, etc.
3. **Click the "Predict Satisfaction" button** to get the satisfaction prediction.
