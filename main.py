import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import os

# function to fetch and process crypto data
def fetch_and_process_crypto_data(coin='ethereum', days=365, api_key="CG-JDG6gBA32v3WUqWnhPUT7ZLC"):

    # API request
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    response = requests.get(url, params=params)

    # check for errors
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    # convert API response to DataFrame
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price']).merge(
        pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume']), on='timestamp'
    )
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    # feature engineering
    df['ma7'] = df['price'].rolling(window=7).mean()
    df['ma30'] = df['price'].rolling(window=30).mean()
    df['volatility'] = df['price'].rolling(window=7).std()
    df['price_change'] = df['price'].pct_change()

    # drop rows with NaN values
    return df.dropna()

# fetch and process data
crypto_data = fetch_and_process_crypto_data(api_key="CG-JDG6gBA32v3WUqWnhPUT7ZLC")
if crypto_data is None:
    exit()

# scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(crypto_data[['price', 'volume', 'ma7', 'ma30', 'volatility', 'price_change']])

# generate sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predicting 'price'
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# split data
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# build model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    shuffle=False
)

# save model
model.save("models/lstm_model.keras")

# evaluate model
predictions = model.predict(X_test)

# reverse scaling
predicted_prices = scaler.inverse_transform(
    np.hstack([predictions, np.zeros((predictions.shape[0], 5))])
)[:, 0]
actual_prices = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 5))])
)[:, 0]

# visualize training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# visualize predictions
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label="Actual Prices", color='blue')
plt.plot(predicted_prices, label="Predicted Prices", color='orange')
plt.title("Predicted vs Actual Ethereum Prices")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

#next day's price
def predict_tomorrow(model, recent_data, scaler):
    """
    Predict if Ethereum's price will go up or down tomorrow.
    """
    # use the last 60 days as input
    last_sequence = recent_data[-60:]  # Last 60 days of scaled data
    last_sequence = np.expand_dims(last_sequence, axis=0)  # Reshape for LSTM input

    # predict the next day's price
    predicted_price_scaled = model.predict(last_sequence)[0][0]

    # reverse scaling for predicted price
    predicted_price = scaler.inverse_transform(
        np.hstack([[predicted_price_scaled], [0], [0], [0], [0], [0]]).reshape(1, -1)
    )[0][0]

    # get today's actual price
    today_price = scaler.inverse_transform(
        np.hstack([recent_data[-1][0], 0, 0, 0, 0, 0]).reshape(1, -1)
    )[0][0]

    # determine if the price will go up or down
    direction = "up" if predicted_price > today_price else "down"

    return direction, today_price, predicted_price