# Chennai Reservoir Water Level Predictor

A simple Flask web app that predicts the total water level of Chennai's four major reservoirs (Poondi, Cholavaram, Redhills, Chembarambakkam) based on rainfall input. Built as a beginner-friendly machine learning + web app project.

Features


# 🔐 Simple login system (checks credentials against users.csv)
# 🌧️ Enter rainfall data for the 4 reservoirs
# 🤖 Predicts total water level using a trained ML model (best_model.pkl) and a feature scaler (scaler.pkl)
# 🖥️ Lightweight — single app.py file, no complex setup


# Project Structure

├── app.py                          # Main Flask application
├── best_model.pkl                  # Trained ML model (Lasso Regression)
├── scaler.pkl                      # Fitted scaler for input features
├── users.csv                       # User login credentials
├── chennai_reservoir_levels.csv    # Historical reservoir water level data
├── chennai_reservoir_rainfall.csv  # Historical rainfall data
└── README.md

# Tech Stack


Python
Flask
Pandas
Scikit-learn


Getting Started

Prerequisites


Python 3.8+
pip




bash   python app.py






# Dataset

The historical data (chennai_reservoir_levels.csv and chennai_reservoir_rainfall.csv) contains daily rainfall and water level readings for the four reservoirs starting from 2004, sourced from Chennai Metropolitan Water Supply and Sewerage Board (CMWSSB) records.

# Notes


This is a learning/demo project — the login system stores plain-text passwords and is not secure for production use.
Make sure best_model.pkl and scaler.pkl are valid and were trained together with matching feature order (Poondi, Cholavaram, Redhills, Chembarambakkam).


# License

This project is open source and available under the MIT License.
