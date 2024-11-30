from flask import Flask, request, render_template
import numpy as np
import pickle
import requests
from datetime import datetime

# Load models
RF1 = pickle.load(open('RF1.pkl','rb'))
preprocessor1 = pickle.load(open('preprocessor1.pkl', 'rb'))
item_list = ['Potatoes','Maize','Wheat','Rice, paddy','Soybeans','Sorghum','Sweet potatoes','Cassava','Yams',
             'Plantains' ]

app = Flask(__name__)

def get_historical_weather(api_key, city_name, start_date, end_date):
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    params = {
        'key': api_key,
        'location': city_name,
        'startDateTime': start_date,
        'endDateTime': end_date,
        'unitGroup': 'metric',
        'contentType': 'json'
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def calculate_weather_data(data):
    if data and 'days' in data:
        total_rainfall = sum(day['precip'] for day in data['days'])
        average_temperature = sum(day['temp'] for day in data['days']) / len(data['days'])
        return total_rainfall, average_temperature

    return None, None


def calculate_average_daily_rainfall(data):
    if data and 'days' in data:
        total_rainfall = sum(day['precip'] for day in data['days'])
        num_days = len(data['days'])
        return total_rainfall / num_days if num_days else 0
    return 0

def predict_yearly_rainfall(average_daily_rainfall):
    # Assuming we're calculating for the remaining days of the current year
    remaining_days = (datetime(datetime.now().year, 12, 31) - datetime.now()).days
    return average_daily_rainfall * remaining_days


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    max_yield = 0;
    if request.method == 'POST':
        # item = request.form['Item']
        pesticides_tonnes = request.form['pesticides_tonnes']
        location = request.form['location']

        # Assume API key and dates are set up here
        api_key = '2Q5CH3KPQ7PCCPKUML4ZP5F46'
        start_date = datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        weather_data = get_historical_weather(api_key, location, start_date, end_date)
        total_rainfall, average_temperature = calculate_weather_data(weather_data)
        average_daily_rainfall = calculate_average_daily_rainfall(weather_data)
        predicted_rainfall = predict_yearly_rainfall(average_daily_rainfall)


        if predicted_rainfall is None or average_temperature is None:
            return render_template('index.html', error="Could not fetch weather data")
        for Item in item_list:
         features = np.array([[Item, total_rainfall, pesticides_tonnes, average_temperature]], dtype=object)
         transformed_features = preprocessor1.transform(features)
         predicted_value = RF1.predict(transformed_features)[0]
         if (predicted_value > max_yield):
             max_yield = predicted_value
             best_item = Item

        return render_template('index.html', predicted_value=max_yield,total_rainfall=predicted_rainfall, average_temperature=average_temperature,best = best_item)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5003,debug = True)