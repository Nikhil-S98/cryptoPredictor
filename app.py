import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# initialize Flask app
app = Flask(__name__)

# load the trained model
model = load_model('models/lstm_model.keras')
scaler = MinMaxScaler()

# function to fetch and process Ethereum data
def fetch_and_process_data():
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    params = {'vs_currency': 'usd', 'days': 365}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price']).merge(
        pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume']), on='timestamp'
    )
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['ma7'] = df['price'].rolling(window=7).mean()
    df['ma30'] = df['price'].rolling(window=30).mean()
    df['volatility'] = df['price'].rolling(window=7).std()
    df['price_change'] = df['price'].pct_change()
    df = df.dropna()
    return df

# function to predict tomorrow's price movement
def predict_tomorrow():
    data = fetch_and_process_data()
    if data is None:
        return "error", 0, 0

    # scale the data
    scaled_data = scaler.fit_transform(data[['price', 'volume', 'ma7', 'ma30', 'volatility', 'price_change']])

    # prepare the last 60 days as input
    last_sequence = np.expand_dims(scaled_data[-60:], axis=0)

    # predict the next day's price
    predicted_price_scaled = model.predict(last_sequence)[0][0]

    # reverse scaling
    predicted_price = scaler.inverse_transform(
        np.hstack([[predicted_price_scaled], [0], [0], [0], [0], [0]]).reshape(1, -1)
    )[0][0]
    today_price = scaler.inverse_transform(
        np.hstack([scaled_data[-1][0], 0, 0, 0, 0, 0]).reshape(1, -1)
    )[0][0]

    # determine direction
    direction = "up" if predicted_price > today_price else "down"

    return direction, today_price, predicted_price

# define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    direction, today_price, predicted_price = predict_tomorrow()
    if direction == "error":
        return jsonify({"error": "Failed to fetch data."}), 500

    return jsonify({
        "today_price": f"${today_price:.2f}",
        "predicted_price": f"${predicted_price:.2f}",
        "direction": direction
    })

# run app
if __name__ == '__main__':
    app.run(debug=True)