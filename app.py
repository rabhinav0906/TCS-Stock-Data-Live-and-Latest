import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("tcs_lstm_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# Load stock data
def load_data():
    data = pd.read_csv("datasets/TCS_stock_history.csv", parse_dates=["Date"])
    data.sort_values("Date", inplace=True)
    return data

data = load_data()

# Streamlit UI
st.set_page_config(layout="wide")  # Set wide layout for better spacing
col1, col2 = st.columns([0.2, 0.8])  # Divide page into columns

# Display logo in the left column
with col1:
    st.image("assets/th.jpg", 
             width=100)

with col2:
    st.title("TCS Stock Price Prediction")
    st.write("Made by Abhinav Rai") 

st.sidebar.subheader("Select an Option")
option = st.sidebar.selectbox("Choose a feature", ["Stock Analysis (3 Graphs)", "Next Day Prediction", "Next Week Prediction"])

# Function to plot graphs
def plot_graphs():
    st.subheader("📊 TCS Stock Analysis")

    # Graph 1: TCS Stock Close Price Over Time
    fig1 = px.line(data, x='Date', y='Close', title='TCS Stock Close Price Over Time',
                   labels={'Close': 'Stock Price', 'Date': 'Year'}, template='plotly_dark')
    fig1.update_traces(line_color='#1f77b4')
    st.plotly_chart(fig1)

    # Graph 2: TCS Stock Volume Per Year
    data['Year'] = data['Date'].dt.year
    volume_per_year = data.groupby('Year')['Volume'].sum()
    fig2 = px.bar(x=volume_per_year.index, y=volume_per_year.values, 
                  text=[f"{volume:,}" for volume in volume_per_year],
                  labels={'x': 'Years', 'y': 'Stock Volume'}, color=volume_per_year.values,
                  title="TCS Stock Volume Per Year", template="plotly_dark")
    fig2.update_traces(textposition='auto', hoverinfo="text+y")
    st.plotly_chart(fig2)

    # Graph 3: Moving Average of Volume
    data['Moving_Avg_Volume'] = data['Volume'].rolling(window=100).mean()
    fig3 = px.line(data, x='Date', y='Moving_Avg_Volume', title='Moving Average of Volume (100 Records)',
                   labels={'Moving_Avg_Volume': 'Moving Average', 'Date': 'Years'}, template='plotly_dark')
    fig3.update_traces(line_color='#9467bd')
    st.plotly_chart(fig3)

# Function to predict next day price
def predict_next_day():
    seq_length = 60
    last_60_days = data['Close'].values[-seq_length:]
    scaled_input = scaler.transform(last_60_days.reshape(-1, 1)).reshape(1, seq_length, 1)
    predicted_price = model.predict(scaled_input)
    next_day_price = scaler.inverse_transform(predicted_price)[0, 0]
    return next_day_price

# Function to predict next week's prices
def predict_next_week():
    seq_length = 60
    input_seq = data['Close'].values[-seq_length:].reshape(-1, 1)
    predicted_prices = []

    for _ in range(7):
        scaled_input = scaler.transform(input_seq).reshape(1, seq_length, 1)
        predicted_price = model.predict(scaled_input)
        predicted_price_original = scaler.inverse_transform(predicted_price)[0, 0]
        predicted_prices.append(predicted_price_original)
        input_seq = np.append(input_seq[1:], predicted_price_original).reshape(-1, 1)

    return predicted_prices

# Display selected option
if option == "Stock Analysis (3 Graphs)":
    plot_graphs()
elif option == "Next Day Prediction":
    predicted_price = predict_next_day()
    st.subheader("📈 Predicted Next Day Price")
    st.markdown(f"### 💰 **₹{predicted_price:.2f}**")
elif option == "Next Week Prediction":
    predicted_prices = predict_next_week()
    days = [f"Day {i+1}" for i in range(7)]
    fig_week = px.line(x=days, y=predicted_prices, markers=True, 
                       title="📅 Stock Price Prediction for Next Week", 
                       labels={"x": "Days", "y": "Price"}, template='plotly_dark')
    st.plotly_chart(fig_week)
