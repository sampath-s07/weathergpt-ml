import json
import os
from datetime import datetime
from typing import Any, Callable

import requests

WMO = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫"),
    48: ("Icy fog", "🌫"),
    51: ("Light drizzle", "🌦"),
    53: ("Drizzle", "🌦"),
    55: ("Heavy drizzle", "🌧"),
    61: ("Light rain", "🌧"),
    63: ("Rain", "🌧"),
    65: ("Heavy rain", "🌧"),
    71: ("Light snow", "❄️"),
    73: ("Snow", "❄️"),
    75: ("Heavy snow", "❄️"),
    80: ("Rain showers", "🌦"),
    81: ("Rain showers", "🌧"),
    82: ("Violent rain showers", "⛈"),
    85: ("Snow showers", "❄️"),
    95: ("Thunderstorm", "⛈"),
    96: ("Thunderstorm with hail", "⛈"),
    99: ("Severe thunderstorm", "⛈"),
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_live_weather",
            "description": "Get live current conditions and a 7-day forecast for any city worldwide.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name, e.g. London"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_climate",
            "description": "Get historical climate stats from the WeatherGPT dataset for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_rain",
            "description": "Predict rain probability using the trained XGBoost model for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "month": {"type": "integer", "description": "Month number 1-12. Defaults to current month."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_extreme_weather",
            "description": "Check for unusual or dangerous weather using Isolation Forest plus live conditions.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ml_forecast",
            "description": "Get a 7-day temperature forecast from the trained LSTM model using historical city data.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def _anthropic_tools():
    tools = []
    for item in OPENAI_TOOLS:
        fn = item["function"]
        tools.append(
            {
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            }
        )
    return tools


def _wmo(code: int):
    return WMO.get(int(code), ("Unknown conditions", "🌤"))


def _weekday(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
    except Exception:
        return date_str


class ChatEngine:
    def __init__(self, tools: dict[str, Callable[..., Any]]):
        self.tools = tools

    def provider_info(self) -> dict:
        groq = os.getenv("GROQ_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        anthropic = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if groq:
            return {
                "provider": "groq",
                "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                "ready": True,
            }
        if anthropic:
            return {
                "provider": "anthropic",
                "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                "ready": True,
            }
        if openai_key:
            return {
                "provider": "openai",
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "ready": True,
            }
        return {"provider": None, "model": None, "ready": False}

    def system_prompt(self) -> str:
        today = datetime.now().strftime("%A, %B %d, %Y")
        return (
            "You are WeatherGPT, a warm, sharp weather assistant that talks like Claude or ChatGPT: "
            "natural, specific, and helpful — never like a keyword bot.\n"
            f"Today is {today}.\n"
            "You have TWO data sources. Use both when a city is involved:\n"
            "1) get_live_weather — current conditions and the 7-day forecast.\n"
            "2) get_city_climate — THIS PROJECT'S historical weather dataset (millions of station records). "
            "Use it to say what is typical for that city: average temp, rain frequency, wind, min/max.\n"
            "Always compare live weather vs the dataset when climate data is found, e.g. "
            "'It is **29°C** now, which is warmer than this city's historical average of **24°C**.'\n"
            "Use predict_rain (XGBoost trained on the dataset), check_extreme_weather "
            "(Isolation Forest trained on the dataset), and get_ml_forecast (LSTM trained on the dataset) "
            "when the user asks about rain chance, danger, or an ML forecast.\n"
            "If the dataset has no match, say so and still answer from live weather.\n"
            "Do not invent numbers when a tool can provide them.\n"
            "If the user omits a city, use a reasonable default from conversation, or ask once.\n"
            "Answer follow-up questions using chat history.\n"
            "Write in short paragraphs. Use markdown **bold** for key numbers. "
            "Give practical advice (clothing, travel, outdoor plans) when it helps.\n"
            "You can chat casually, but stay useful. If something is unrelated to weather, answer briefly then offer weather help."
        )

    def run_tool(self, name: str, args: dict, ui: dict) -> str:
        fn = self.tools.get(name)
        if not fn:
            return json.dumps({"error": f"Unknown tool {name}"})
        try:
            result = fn(**args)
        except TypeError:
            result = fn(args.get("city", ""))
        except Exception as e:
            result = {"error": str(e)}
        if not isinstance(result, dict):
            result = {"result": result}
        self._attach_ui(name, result, ui)
        return json.dumps(result, default=str)

    def _attach_ui(self, name: str, result: dict, ui: dict) -> None:
        if result.get("error") or result.get("found") is False:
            return
        if name == "get_live_weather":
            current = result.get("current") or {}
            ui["card"] = {
                "title": f"Now — {result.get('city', '')}",
                "value": f"{current.get('temp_c', '—')}°C",
                "emoji": current.get("emoji", "🌤"),
                "desc": current.get("summary", ""),
                "prob": current.get("humidity"),
            }
            if result.get("forecast"):
                ui["forecast"] = result["forecast"]
        elif name == "get_city_climate":
            ui["card"] = {
                "title": f"Climate — {result.get('city', '')}",
                "value": f"{result.get('avg_temp', '—')}°C",
                "emoji": "☀️" if float(result.get("avg_temp") or 0) > 20 else "⛅" if float(result.get("avg_temp") or 0) > 10 else "❄️",
                "desc": (
                    f"Min {result.get('min_temp')}°C — Max {result.get('max_temp')}°C · "
                    f"Wind {result.get('avg_wind_kmh')} km/h · "
                    f"Rain {result.get('rain_days_percent')}% of days"
                ),
                "prob": result.get("rain_days_percent"),
            }
        elif name == "predict_rain":
            prob = result.get("rain_probability", 0)
            ui["card"] = {
                "title": "Rain probability",
                "value": f"{prob}%",
                "emoji": "🌧" if prob > 50 else "⛅" if prob > 30 else "☀️",
                "desc": result.get("prediction", ""),
                "prob": prob,
            }
        elif name == "check_extreme_weather":
            alerts = result.get("alerts") or []
            ui["card"] = {
                "title": "Weather safety",
                "value": "Alert" if alerts else "All clear",
                "emoji": "⚠️" if alerts else "🛡️",
                "desc": "; ".join(alerts) if alerts else "No dangerous pattern detected.",
            }
        elif name == "get_ml_forecast" and result.get("forecast"):
            ui["forecast"] = result["forecast"]

    def chat(self, messages: list[dict]) -> dict:
        info = self.provider_info()
        if not info["ready"]:
            return {
                "reply": (
                    "I need a real language model key to chat like Claude or ChatGPT.\n\n"
                    "Easiest free option: create a Groq key at https://console.groq.com/keys "
                    "and put it in a `.env` file in the project folder:\n\n"
                    "GROQ_API_KEY=your_key_here\n\n"
                    "Or use Claude with `ANTHROPIC_API_KEY=...` or ChatGPT with `OPENAI_API_KEY=...`.\n"
                    "Then restart the API server."
                ),
                "provider": None,
                "model": None,
                "card": None,
                "forecast": None,
                "needs_key": True,
            }

        history = []
        for msg in messages[-20:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                history.append({"role": role, "content": content})
        if not history:
            return {"reply": "Ask me anything about the weather.", "provider": info["provider"], "model": info["model"], "card": None, "forecast": None}

        ui = {"card": None, "forecast": None}
        if info["provider"] == "anthropic":
            reply = self._chat_anthropic(history, ui)
        else:
            reply = self._chat_openai_compatible(history, ui, info)
        return {
            "reply": reply,
            "provider": info["provider"],
            "model": info["model"],
            "card": ui["card"],
            "forecast": ui["forecast"],
            "needs_key": False,
        }

    def _chat_openai_compatible(self, history: list[dict], ui: dict, info: dict) -> str:
        if info["provider"] == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            base = "https://api.groq.com/openai/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

        messages = [{"role": "system", "content": self.system_prompt()}] + history
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for _ in range(5):
            payload = {
                "model": info["model"],
                "messages": messages,
                "tools": OPENAI_TOOLS,
                "tool_choice": "auto",
                "temperature": 0.6,
            }
            res = requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=90)
            if res.status_code >= 400:
                raise RuntimeError(f"{info['provider']} error {res.status_code}: {res.text[:500]}")
            choice = res.json()["choices"][0]["message"]
            tool_calls = choice.get("tool_calls") or []
            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.get("content") or "",
                        "tool_calls": tool_calls,
                    }
                )
                for call in tool_calls:
                    fn = call.get("function") or {}
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except json.JSONDecodeError:
                        args = {}
                    output = self.run_tool(fn.get("name", ""), args, ui)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": output,
                        }
                    )
                continue
            return (choice.get("content") or "").strip() or "I could not generate a reply."
        return "I got stuck calling weather tools. Please try again."

    def _chat_anthropic(self, history: list[dict], ui: dict) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        messages = [{"role": m["role"], "content": m["content"]} for m in history]

        for _ in range(5):
            payload = {
                "model": model,
                "max_tokens": 1200,
                "system": self.system_prompt(),
                "tools": _anthropic_tools(),
                "messages": messages,
            }
            res = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=90)
            if res.status_code >= 400:
                raise RuntimeError(f"Anthropic error {res.status_code}: {res.text[:500]}")
            data = res.json()
            blocks = data.get("content") or []
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            if data.get("stop_reason") == "tool_use" and tool_uses:
                messages.append({"role": "assistant", "content": blocks})
                tool_results = []
                for tu in tool_uses:
                    output = self.run_tool(tu.get("name", ""), tu.get("input") or {}, ui)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.get("id"),
                            "content": output,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue
            return "\n".join(t for t in texts if t).strip() or "I could not generate a reply."
        return "I got stuck calling weather tools. Please try again."


def fetch_live_weather(city: str) -> dict:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en"},
        timeout=12,
    )
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        return {"found": False, "message": f"Could not find '{city}' on the live weather map."}
    loc = results[0]
    label = ", ".join(p for p in [loc.get("name"), loc.get("admin1"), loc.get("country")] if p)
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=12,
    )
    weather.raise_for_status()
    data = weather.json()
    cur = data.get("current") or {}
    desc, emoji = _wmo(cur.get("weather_code") or 0)
    daily = data.get("daily") or {}
    forecast = []
    times = daily.get("time") or []
    for i, day in enumerate(times):
        code = (daily.get("weather_code") or [0])[i]
        d, em = _wmo(code)
        tmax = (daily.get("temperature_2m_max") or [None])[i]
        tmin = (daily.get("temperature_2m_min") or [None])[i]
        rain_p = (daily.get("precipitation_probability_max") or [None])[i]
        forecast.append(
            {
                "day": _weekday(day),
                "icon": em,
                "temp": f"{round(tmax)}° / {round(tmin)}°" if tmax is not None and tmin is not None else "—",
                "rain": f"{rain_p}% rain" if rain_p is not None else d,
            }
        )
    return {
        "found": True,
        "city": label,
        "current": {
            "temp_c": cur.get("temperature_2m"),
            "feels_like_c": cur.get("apparent_temperature"),
            "humidity": cur.get("relative_humidity_2m"),
            "precipitation_mm": cur.get("precipitation"),
            "wind_kmh": cur.get("wind_speed_10m"),
            "condition": desc,
            "emoji": emoji,
            "summary": (
                f"{desc}. Feels like {cur.get('apparent_temperature')}°C · "
                f"Humidity {cur.get('relative_humidity_2m')}% · "
                f"Wind {cur.get('wind_speed_10m')} km/h"
            ),
        },
        "forecast": forecast,
        "source": "Open-Meteo live forecast",
    }
