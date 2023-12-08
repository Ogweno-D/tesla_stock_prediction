import pickle
import streamlit as st
import yfinance as yf
import pandas as pd

# Load the trained model
model_path = 'ai_trial.ipynb'
model = pickle.load(open(model_path, 'rb'))

def main():
    st.title('Stock Prediction App')

    # Input Variables
    stock_name = st.text_input('Enter Stock Symbol (e.g., TSLA):', 'TSLA')
    start_date = st.date_input('Select Start Date:', pd.to_datetime('2022-01-01'))
    end_date = st.date_input('Select End Date:', pd.to_datetime('2023-01-01'))

    if st.button('Predict'):
        # Download historical stock data
        stock_data = yf.download(stock_name, start=start_date, end=end_date)

        # Preprocess data if needed
        # ...

        # Make predictions using the loaded model
        # Replace this with your actual feature extraction and prediction code
        predictions = model.predict(stock_data[['Open', 'High', 'Low', 'Volume']])

        # Display predictions
        st.subheader('Predicted Stock Prices:')
        st.line_chart(predictions)

if __name__ == '__main__':
    main()
