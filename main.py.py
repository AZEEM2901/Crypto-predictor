from fastapi import FastAPI, Query
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time
import os

app = FastAPI()

# Cache dictionary and TTL (time to live) in seconds
cache = {}
CACHE_TTL = 25 * 60  # 25 minutes

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, nhead=2):
        super(HybridModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        transformer_input = gru_out.permute(1, 0, 2)
        transformer_out = self.transformer(transformer_input)
        output = self.fc(transformer_out[-1])
        return output

@app.get("/")
def home():
    return {"message": "Welcome to Crypto Predictor API. Use /predict?coin=bitcoin"}

@app.get("/predict")
def predict(coin: str = Query(None, description="Cryptocurrency coin, e.g. bitcoin, ethereum, solana, cardano")):
    if not coin:
        return {"error": "Please specify a coin using the 'coin' query parameter, e.g. /predict?coin=bitcoin"}

    now = time.time()
    # Check cache for requested coin
    if coin in cache and now - cache[coin]['timestamp'] < CACHE_TTL:
        return cache[coin]['data']

    try:
        # CoinCap expects coin IDs like 'bitcoin', 'ethereum', etc.
        url = f"https://api.coincap.io/v2/assets/{coin}/history?interval=m1"
        response = requests.get(url)

        if response.status_code != 200:
            return {"error": f"Failed to fetch data: {response.text}"}

        data = response.json().get("data", [])
        if len(data) < 60:
            return {"error": "Not enough data to predict"}

        # CoinCap returns list of dict with "priceUsd" and "time"
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time"], unit='ms')
        df["price"] = df["priceUsd"].astype(float)

        prices = df["price"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices)

        last_seq = scaled[-30:]
        X = torch.tensor(last_seq.reshape(1, 30, 1), dtype=torch.float32)

        model = HybridModel(input_dim=1, hidden_dim=64, output_dim=1)
        model.eval()

        predictions = []
        with torch.no_grad():
            current_seq = X.clone()
            for _ in range(30):  # predict 30 mins ahead
                pred = model(current_seq).item()
                predictions.append(pred)
                new_input = torch.tensor([[[pred]]], dtype=torch.float32)
                current_seq = torch.cat((current_seq[:, 1:, :], new_input), dim=1)

        predicted_price = scaler.inverse_transform(np.array([[predictions[-1]]])).flatten()[0]
        current_price = df["price"].iloc[-1]
        last_time = df["timestamp"].iloc[-1]
        pred_time = last_time + timedelta(minutes=30)

        result = {
            "coin": coin,
            "current_price": round(current_price, 2),
            "predicted_price_30min": round(predicted_price, 2),
            "prediction_time": pred_time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save to cache
        cache[coin] = {
            "timestamp": now,
            "data": result
        }

        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
