## Stock Price Prediction using LSTM

Project Overview

This project involves building a Long Short-Term Memory (LSTM) model, a type of Recurrent Neural Network (RNN), to predict stock prices using historical stock data. The LSTM model is well-suited for this task due to its ability to capture long-term dependencies in time series data, such as stock prices.

The model was trained on historical stock price data and aims to predict the future stock prices over a specific time window. It also involves exploring data preprocessing techniques like First-Order Differencing to make the time series stationary, which helps improve the accuracy of predictions.

The notebook is freely available, and I hope it serves as a helpful resource for others interested in time series forecasting using LSTMs.
Features

Time Series Forecasting: Prediction of stock prices using LSTM.
Model Evaluation: Evaluation metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
Visualizations: Plotting actual vs. predicted values, future predictions, and training loss over epochs.
Data Preprocessing: Data cleaning, normalization using MinMaxScaler, and handling outliers.

Key Technologies

Python: Core programming language used.
TensorFlow: Deep learning framework used to build and train the LSTM model.
NumPy & Pandas: Data manipulation and processing.
Matplotlib: For visualizations.
Scikit-Learn: For data preprocessing tasks like scaling and train-test splitting.

Dataset

The dataset used for training is historical stock price data. It contains features such as:

Date: The date the stock price data was recorded.
Open, High, Low, Close: The stock prices at different times during the trading day.
Volume: The total number of shares traded on that day.

The dataset is cleaned and preprocessed to focus primarily on the 'Close' price for prediction.
Setup Instructions

To run this project locally, follow these steps:
Prerequisites

Python 3.7 or above
Install the required libraries:

  ``bash
  pip install numpy pandas matplotlib tensorflow scikit-learn
  ``

Running the Code

Clone this repository:
  ``bash
  git clone https://github.com/dulhara19/ML-project07-LSTM-stock_price_prediction.git
  cd stock-price-prediction
  ``
