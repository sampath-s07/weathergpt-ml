import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
from dotenv import load_dotenv

from api.chat_engine import ChatEngine, fetch_live_weather

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load NLP keywords
with open(os.path.join(BASE_DIR, 'models', 'nlp', 'intent_keywords.json'), 'r') as f:
    keywords = json.load(f)

# Load dataset
try:
    cols = ['city_name', 'avg_temp_c', 'min_temp_c', 'max_temp_c',
        'precipitation_mm', 'will_rain', 'avg_wind_speed_kmh']
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'weather_clean.csv'),
                 usecols=cols, low_memory=False)

    df['city_name'] = df['city_name'].str.lower().str.strip()
    print(f"Dataset loaded: {len(df):,} rows")
except Exception as e:
    df = None
    print(f"Dataset not loaded: {e}")

# Load XGBoost rain model
try:
    xgb_model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost', 'model.pkl'))
    print("XGBoost model loaded!")
except Exception as e:
    xgb_model = None
    print(f"XGBoost not loaded: {e}")

# Load XGBoost scaler
try:
    xgb_scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost', 'scaler.pkl'))
    print("XGBoost scaler loaded!")
except:
    xgb_scaler = None

# Load Isolation Forest
try:
    iso_model = joblib.load(os.path.join(BASE_DIR, 'models', 'anomaly', 'model.pkl'))
    print("Isolation Forest loaded!")
except Exception as e:
    iso_model = None
    print(f"Isolation Forest not loaded: {e}")

# Load LSTM
try:
    import tensorflow as tf
    lstm_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'lstm', 'model.h5'), compile=False)
    print("LSTM model loaded!")
except Exception as e:
    lstm_model = None
    print(f"LSTM not loaded: {e}")

# Load LSTM scaler
try:
    lstm_scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'lstm', 'scaler.pkl'))
    print("LSTM scaler loaded!")
except:
    lstm_scaler = None
def classify_intent(text):
    text = text.lower()
    if any(w in text for w in keywords['rain_words']):
        return 'rain_prediction'
    elif any(w in text for w in keywords['extreme_words']):
        return 'extreme_weather'
    elif any(w in text for w in keywords['forecast_words']):
        return 'forecast'
    elif any(w in text for w in keywords['temperature_words']):
        return 'temperature'
    elif any(w in text for w in keywords['activity_words']):
        return 'activity_advice'
    return 'general'

def extract_city(text):
    stop_words = ['weather', 'in', 'for', 'at', 'the', 'what', 'is',
                  'tell', 'me', 'about', 'temperature', 'rain', 'forecast',
                  'today', 'tomorrow', 'will', 'it', 'how', 'hot', 'cold',
                  'show', 'give', 'get', 'city', 'place', 'any', 'there',
                  'coming', 'outside', 'safe', 'week', 'this', 'good']
    words = text.lower().split()
    candidates = [w for w in words if w not in stop_words and len(w) > 2]
    return candidates[0] if candidates else None

def get_city_weather(city_name):
    if df is None:
        return None
    city_data = df[df['city_name'].str.contains(city_name.lower(), na=False)]
    if city_data.empty:
        return None
    return {
        'city': city_name.title(),
        'avg_temp': round(float(city_data['avg_temp_c'].mean()), 1),
        'min_temp': round(float(city_data['min_temp_c'].mean()), 1),
        'max_temp': round(float(city_data['max_temp_c'].mean()), 1),
        'avg_rain_mm': round(float(city_data['precipitation_mm'].mean()), 1),
        'rain_days_percent': round(float(city_data['will_rain'].mean()) * 100, 1),
        'avg_wind_kmh': round(float(city_data['avg_wind_speed_kmh'].mean()), 1),
        'total_records': len(city_data)
    }

def predict_rain_from_model(month, temp, min_temp, max_temp, wind, rain_mm=0):
    # Use real XGBoost model if loaded
    if xgb_model is not None:
        try:
            features = np.array([[month, temp, min_temp, max_temp, wind, rain_mm]])
            if xgb_scaler is not None:
                features = xgb_scaler.transform(features)
            prob = xgb_model.predict_proba(features)[0][1]
            return round(float(prob) * 100, 1)
        except Exception as e:
            print(f"XGBoost predict error: {e}")
    # Fallback calculation using dataset patterns
    score = 0
    if month in [11, 12, 1, 2, 3]: score += 30
    if month in [4, 5, 10]: score += 15
    if temp < 10: score += 20
    if temp < 5: score += 10
    if wind > 25: score += 15
    if wind > 40: score += 10
    if rain_mm > 0: score += 15
    if max_temp - min_temp > 12: score += 10
    return min(int(score), 95)

def predict_extreme_from_model(temp, wind, rain_mm):
    alerts = []
    # Use real Isolation Forest if loaded
    if iso_model is not None:
        try:
            features = np.array([[temp, temp - 5, temp + 5, wind, rain_mm]])
            result = iso_model.predict(features)
            if result[0] == -1:
                if temp > 38: alerts.append("Extreme heat warning!")
                elif temp < -10: alerts.append("Extreme cold warning!")
                elif wind > 75: alerts.append("Severe storm warning!")
                elif rain_mm > 80: alerts.append("Flood risk warning!")
                else: alerts.append("Unusual weather pattern detected!")
        except Exception as e:
            print(f"Isolation Forest predict error: {e}")
    # Also check thresholds regardless
    if temp > 40 and "Extreme heat" not in str(alerts): alerts.append("Extreme heat warning!")
    if temp < -15 and "Extreme cold" not in str(alerts): alerts.append("Extreme cold warning!")
    if wind > 80 and "storm" not in str(alerts): alerts.append("Severe storm warning!")
    if rain_mm > 100 and "Flood" not in str(alerts): alerts.append("Flood risk warning!")
    return alerts

def get_lstm_forecast(city_name=None):
    # Try to get real LSTM forecast
    if lstm_model is not None and df is not None:
        try:
            if city_name:
                city_data = df[df['city_name'].str.contains(city_name.lower(), na=False)]
            else:
                city_data = df[df['city_name'].str.contains('brussels', na=False)]
            if len(city_data) >= 30:
                last_30 = city_data['avg_temp_c'].tail(30).values.reshape(-1, 1)
                if lstm_scaler is not None:
                    last_30_scaled = lstm_scaler.transform(last_30)
                else:
                    last_30_scaled = (last_30 - last_30.mean()) / (last_30.std() + 1e-8)
                X = last_30_scaled.reshape(1, 30, 1)
                pred = lstm_model.predict(X, verbose=0)
                if lstm_scaler is not None:
                    temps = lstm_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                else:
                    temps = pred.flatten() * (last_30.std() + 1e-8) + last_30.mean()
                return [round(float(t), 1) for t in temps[:7]]
        except Exception as e:
            print(f"LSTM forecast error: {e}")
    return None

def get_city_recent_conditions(city_name):
    if df is None:
        return None
    try:
        city_data = df[df['city_name'].str.contains(city_name.lower(), na=False)]
        if city_data.empty:
            return None
        recent = city_data.tail(30)
        return {
            'avg_temp': round(float(recent['avg_temp_c'].mean()), 1),
            'min_temp': round(float(recent['min_temp_c'].mean()), 1),
            'max_temp': round(float(recent['max_temp_c'].mean()), 1),
            'avg_wind': round(float(recent['avg_wind_speed_kmh'].mean()), 1),
            'avg_rain_mm': round(float(recent['precipitation_mm'].mean()), 1),
            'rain_pct': round(float(recent['will_rain'].mean()) * 100, 1)
        }
    except:
        return None
def tool_rain(city: str, month: int = None):
    if month is None:
        month = datetime.now().month
    data = predict_rain(month=int(month), city=city)
    data["city"] = city
    return data


def tool_extreme(city: str):
    live = fetch_live_weather(city)
    if live.get("found"):
        cur = live["current"]
        data = predict_extreme(
            temp=float(cur.get("temp_c") or 12),
            wind=float(cur.get("wind_kmh") or 18),
            rain=float(cur.get("precipitation_mm") or 0),
            city=city,
        )
        data["live_city"] = live.get("city")
        data["live_conditions"] = cur
        return data
    return predict_extreme(city=city)


def tool_ml_forecast(city: str):
    raw = forecast(city=city)
    days = []
    for d in raw.get("forecast") or []:
        temp = d.get("temp")
        days.append({
            "day": str(d.get("day", ""))[:3],
            "icon": d.get("icon", "🌤"),
            "temp": temp if isinstance(temp, str) else f"{temp}°C",
            "rain": d.get("rain", raw.get("model_used", "")),
        })
    return {"city": raw.get("city"), "forecast": days, "model_used": raw.get("model_used")}


def tool_climate(city: str):
    data = get_city_weather(city)
    if not data:
        return {"found": False, "message": f"No historical dataset match for '{city}'."}
    return data


chat_engine = ChatEngine({
    "get_live_weather": fetch_live_weather,
    "get_city_climate": tool_climate,
    "predict_rain": tool_rain,
    "check_extreme_weather": tool_extreme,
    "get_ml_forecast": tool_ml_forecast,
})


class ChatRequest(BaseModel):
    messages: list[dict]


@app.get("/")
def home():
    return {
        "status": "WeatherGPT API running",
        "dataset_rows": len(df) if df is not None else 0,
        "models": {
            "xgboost": xgb_model is not None,
            "isolation_forest": iso_model is not None,
            "lstm": lstm_model is not None
        },
        "llm": chat_engine.provider_info()
    }


@app.get("/ui")
def ui():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return chat_engine.chat(req.messages)
    except Exception as e:
        return {
            "reply": f"The language model request failed:\n\n{e}",
            "provider": None,
            "model": None,
            "card": None,
            "forecast": None,
            "needs_key": False,
        }

@app.get("/ask")
def ask(question: str):
    intent = classify_intent(question)
    city = extract_city(question)
    city_data = None
    if city:
        city_data = get_city_weather(city)
    return {
        "question": question,
        "intent": intent,
        "city": city,
        "city_data": city_data,
        "status": "success"
    }

@app.get("/city")
def city_weather(city: str):
    data = get_city_weather(city)
    if data:
        return {"found": True, "weather": data}
    return {"found": False, "message": f"City '{city}' not found in dataset"}

@app.get("/predict/rain")
def predict_rain(month: int = 10, temp: float = 12.0,
                 min_temp: float = 8.0, max_temp: float = 16.0,
                 wind: float = 18.0, rain_mm: float = 0.0,
                 city: str = None):
    # Use real city data if provided
    if city and df is not None:
        conditions = get_city_recent_conditions(city)
        if conditions:
            temp = conditions['avg_temp']
            min_temp = conditions['min_temp']
            max_temp = conditions['max_temp']
            wind = conditions['avg_wind']
            rain_mm = conditions['avg_rain_mm']

    prob = predict_rain_from_model(month, temp, min_temp, max_temp, wind, rain_mm)
    return {
        "rain_probability": prob,
        "prediction": "rain" if prob > 50 else "no rain",
        "conditions_used": {
            "month": month,
            "temp": temp,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "wind_kmh": wind,
            "rain_mm": rain_mm
        },
        "model_used": "XGBoost" if xgb_model is not None else "rule_based"
    }

@app.get("/predict/extreme")
def predict_extreme(temp: float = 12.0, wind: float = 18.0,
                    rain: float = 0.0, city: str = None):
    # Use real city data if provided
    if city and df is not None:
        conditions = get_city_recent_conditions(city)
        if conditions:
            temp = conditions['avg_temp']
            wind = conditions['avg_wind']
            rain = conditions['avg_rain_mm']

    alerts = predict_extreme_from_model(temp, wind, rain)
    return {
        "is_extreme": len(alerts) > 0,
        "alerts": alerts,
        "conditions_checked": {
            "temp": temp,
            "wind_kmh": wind,
            "rain_mm": rain
        },
        "model_used": "IsolationForest" if iso_model is not None else "threshold_based"
    }

@app.get("/forecast")
def forecast(city: str = "brussels"):
    temps = get_lstm_forecast(city)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    icons = ["🌤", "⛅", "🌧", "🌤", "☀️", "⛅", "🌤"]

    if temps and len(temps) >= 7:
        forecast_data = []
        for i in range(7):
            t = temps[i]
            icon = "☀️" if t > 20 else "🌤" if t > 15 else "⛅" if t > 10 else "🌧" if t > 5 else "❄️"
            forecast_data.append({
                "day": days[i],
                "temp": round(t, 1),
                "icon": icon
            })
        return {
            "city": city.title(),
            "forecast": forecast_data,
            "model_used": "LSTM"
        }
    else:
        # Fallback using dataset averages
        if df is not None:
            try:
                city_data = df[df['city_name'].str.contains(city.lower(), na=False)]
                if not city_data.empty:
                    base_temp = float(city_data['avg_temp_c'].mean())
                    forecast_data = []
                    for i in range(7):
                        variation = round(float(np.random.uniform(-3, 3)), 1)
                        t = round(base_temp + variation, 1)
                        icon = "☀️" if t > 20 else "🌤" if t > 15 else "⛅" if t > 10 else "🌧" if t > 5 else "❄️"
                        forecast_data.append({
                            "day": days[i],
                            "temp": t,
                            "icon": icon
                        })
                    return {
                        "city": city.title(),
                        "forecast": forecast_data,
                        "model_used": "dataset_average"
                    }
            except:
                pass
        return {
            "city": city.title(),
            "forecast": [{"day": days[i], "temp": 12.0, "icon": icons[i]} for i in range(7)],
            "model_used": "fallback"
        }
