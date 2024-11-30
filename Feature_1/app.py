
from flask import Flask,request, render_template
import numpy as np
import pickle


#loading models
RF1 = pickle.load(open('RF1.pkl','rb'))
preprocessor1 = pickle.load(open('preprocessor1.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    if request. method == 'POST':
        Item  = request.form['Item']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']

        features = np.array([[Item,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]],dtype=object)
        transformed_features = preprocessor1.transform(features)
        predicted_value = RF1.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',predicted_value =predicted_value)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001,debug = True)