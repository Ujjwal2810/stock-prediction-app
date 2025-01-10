import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Custom CSS to improve the layout and design
st.markdown("""
    <style>
        .main { background-color: #f4f4f9; padding: 30px; }
        h1 { color: #1f77b4; font-size: 36px; font-family: 'Arial', sans-serif; }
        h2 { color: #ff7f0e; font-size: 28px; font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background-color: #333; color: white; }
        .stTextInput, .stDateInput { font-size: 18px; padding: 10px; }
        .stButton { background-color: #1f77b4; color: white; font-size: 16px; padding: 10px; }
        .stPlotlyChart { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model('E:/jn/Stock Predictions Model.keras')

# Sidebar input options
st.sidebar.header('Stock Prediction Settings')
stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2012-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2024-12-31'))

# Fetch stock data
data = yf.download(stock, start=start_date, end=end_date)

# Display stock data
st.title(f"Stock Prediction for {stock}")
st.write("Displaying historical stock data:")
st.write(data)

# Prepare the data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# Add charts with better labels
st.subheader('Stock Price vs Moving Averages (50 Days)', anchor="ma50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(12, 7))
plt.plot(ma_50_days, 'r', label='50-Day MA', linewidth=2)
plt.plot(data.Close, 'g', label='Stock Price', alpha=0.7)
plt.title(f'{stock} Price vs MA50', fontsize=18)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Stock Price vs Moving Averages (50, 100 Days)', anchor="ma100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(12, 7))
plt.plot(ma_50_days, 'r', label='50-Day MA', linewidth=2)
plt.plot(ma_100_days, 'b', label='100-Day MA', linewidth=2)
plt.plot(data.Close, 'g', label='Stock Price', alpha=0.7)
plt.title(f'{stock} Price vs MA50 & MA100', fontsize=18)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Stock Price vs Moving Averages (50, 100, 200 Days)', anchor="ma200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12, 7))
plt.plot(ma_100_days, 'r', label='100-Day MA', linewidth=2)
plt.plot(ma_200_days, 'b', label='200-Day MA', linewidth=2)
plt.plot(data.Close, 'g', label='Stock Price', alpha=0.7)
plt.title(f'{stock} Price vs MA100 & MA200', fontsize=18)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Model prediction
predicted_price = model.predict(x)
scale = 1 / scaler.scale_
predicted_price = predicted_price * scale
y = y * scale

# Display prediction vs original price
st.subheader('Stock Price Prediction vs Actual Price', anchor="prediction")
fig4 = plt.figure(figsize=(12, 7))
plt.plot(predicted_price, 'r', label='Predicted Price', linewidth=2)
plt.plot(y, 'g', label='Actual Price', alpha=0.7)
plt.title(f'{stock} Predicted vs Actual Price', fontsize=18)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Final Note with a Button to Rerun
st.markdown("#### Enjoyed the Prediction? 🔮")
if st.button('Predict Another Stock'):
    st.experimental_rerun()
