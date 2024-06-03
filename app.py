# app.py
from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io
import base64

app = Flask(__name__)

def predict_stock(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    lookback = 50
    X, y = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i:i+lookback, 0])
        y.append(data_scaled[i+lookback, 0])
    X, y = np.array(X), np.array(y)

    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=8)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(8, 4))
    plt.plot(predictions, label='Tahminler', linewidth=1.5)
    plt.plot(y_test, label='Gerçek Değerler', linewidth=1.5)
    plt.legend()
    plt.title(f'{symbol} Hisse Senedi Fiyat Tahmini')
    plt.xlabel('Günler')
    plt.ylabel('Kapanış Fiyatı')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    plot_url = predict_stock(symbol, start_date, end_date)
    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
