from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="weatherGPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, 'models', 'nlp', 'intent_keywords.json'), 'r') as f:
    keywords = json.load(f)

def classify_intent(text):
    text = text.lower()
    if any(w in text for w in keywords['rain_words']):
        return 'rain_prediction'
    elif any(w in text for w in keywords['extreme_words']):
        return 'extreme_weather'
    elif any(w in text for w in keywords['forecast_words']):
        return 'forecast'
    elif any(w in text for w in keywords['temp_words']):
        return 'temperature'
    elif any(w in text for w in keywords['activity_words']):
        return 'activity_advice'
    return 'general'

@app.get("/")
def home():
    return {"message:" "WeatherGPT API is running!"}

@app.get("/ask")
def ask(question: str):
    intent = classify_intent(question)
    return {
        "question": question,
        "intent": intent,
        "status": "success"
    } 

@app.get("/predict/rain")
def predict_rain(month: int, temp: float, min_temp: float, max_temp: float, wind: float):

    score = 0
    if month in [6, 7, 8, 9]:
        score += 40
    if temp < 20:
        score += 20
    if wind > 30:
        score += 20
    if max_temp - min_temp > 15:
        score += 20
    return {
        "rain_probability": min(score, 95),
        "prediction": "rain" if score > 50 else "no rain"
    }

@app.get("/predict/extreme")
def predict_extreme(temp: float, wind: float, rain: float):
    alerts = []
    if temp > 45:
        alerts.append("Extreme heat warning!")
    if temp < -10:
        alerts.append("Extreme cold warning!")
    if wind > 100:
        alerts.append("Storm warning!")
    if rain > 100:
        alerts.append("Flood risk!")
    return {
        "is_extreme": len(alerts) > 0,
        "alerts": alerts if alerts else ["Normal weather conditions"]
    }