import requests
from datetime import datetime


def get_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': 52.0417,
        'longitude': -0.7558,
        'hourly': 'apparent_temperature',
        'forecast_days': 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Get current day's date
    current_date_str = datetime.now().strftime('%Y-%m-%d')

    # Find the index of tomorrow's noon temperature
    for i, time in enumerate(data['hourly']['time']):
        if f"{current_date_str}T12:00" in time:
            return data['hourly']['apparent_temperature'][i]

    # In case we didn't find the specific time, return None
    return None


def main():
    forecast = get_forecast()
    if forecast is not None:
        print(f"Tomorrow's weather forecast at 12:00 is: {forecast} degrees.")
    else:
        print("Could not find tomorrow's weather forecast at 12:00.")


if __name__ == "__main__":
    main()
