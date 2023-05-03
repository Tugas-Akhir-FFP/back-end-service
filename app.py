from flask import Flask, request
import json
from flask_restful import Resource, Api
from flask_cors import CORS
import gspread
import pandas as pd
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from waitress import serve

app = Flask(__name__)
api = Api(app)
CORS(app)

param_grid = {'seasonal': ['additive', 'multiplicative'],
              'trend': ['additive', 'multiplicative'],
              'seasonal_periods': [4, 12],
              'smoothing_level': np.linspace(0.1, 0.9, 9),
              'smoothing_trend': np.linspace(0.1, 0.9, 9),
              'smoothing_seasonal': np.linspace(0.1, 0.9, 9)}

def grid_search(df):
    print(len(df))
    train, test = df[1700:1800], df[1800:1826]
    results = []
    for seasonal in param_grid['seasonal']:
        for trend in param_grid['trend']:
            for seasonal_period in param_grid['seasonal_periods']:
                for smoothing_level in param_grid['smoothing_level']:
                    for smoothing_trend in param_grid['smoothing_trend']:
                        for smoothing_seasonal in param_grid['smoothing_seasonal']:
                            try:
                                model = ExponentialSmoothing(train, 
                                                            seasonal=seasonal, 
                                                            trend=trend, 
                                                            seasonal_periods=seasonal_period)
                                model_fit = model.fit(smoothing_level=smoothing_level, 
                                                    smoothing_trend=smoothing_trend, 
                                                    smoothing_seasonal=smoothing_seasonal)
                                # Make predictions and calculate R-squared
                                predictions = model_fit.predict(start=len(train), end = len(train)+len(test)-1)
    
                                r2 = r2_score(test, predictions)
                                results.append((seasonal, trend, seasonal_period, smoothing_level, smoothing_trend, smoothing_seasonal, r2))
                            except:
                                continue
    
    # Cari parameter yang menghasilkan r-square terbaik
    best_params = max(results, key=lambda x: x[-1])
    model = ExponentialSmoothing(train,
                                 seasonal=best_params[0],
                                 trend=best_params[1],
                                 seasonal_periods=best_params[2])
    model_fit = model.fit(smoothing_level=best_params[3], 
                          smoothing_trend=best_params[4], 
                          smoothing_seasonal=best_params[5])
    predictions = model_fit.forecast(len(test))
    r2 = r2_score(test, predictions)
    mse = mean_squared_error(test, predictions)  
    hasil = {};
    hasil['Hasil'] = predictions.tolist()
    print(r2,'r2')
    print(mse,'mse')
    print(test)
    print(best_params)
    return predictions.tolist() 

def dataProcessing(data, periods, start, end): 
    try:
        data = data[1:]  # menghapus baris label kolom dari data
        df = pd.DataFrame(data, columns=['Tanggal', 'Tavg', 'RH_avg', 'ff_avg', 'RR'])  # hanya memasukkan data
        df = df.rename(columns={'Tanggal': 'Date', 'Tavg': 'Temperature',
                                'RH_avg': 'Humidity', 'ff_avg': 'Wind', 'RR': 'Rainfall'})
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

        # filter data within date range
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        
        # convert columns to float and fill missing values
        df[['Temperature', 'Humidity', 'Wind', 'Rainfall']] = df[['Temperature', 'Humidity', 'Wind', 'Rainfall']].astype(float)
        df[['Temperature', 'Humidity', 'Wind', 'Rainfall']] = df[['Temperature', 'Humidity', 'Wind', 'Rainfall']].fillna(method='ffill')

        # set Date as index and reindex with complete date range
        df = df.set_index('Date')
        df = df.reindex(index, fill_value=np.nan)
        response = {}
        response['Hasil'] = df['Temperature'].tolist()[-periods:]

        return response
    except Exception as e:
        return {'error': str(e)}
# Riau-Kab.Kampar_2015-2019
# Data Harian - Table   
@app.route('/api')
def get_credentials():
    sheetName = request.args.get('sheetName')
    worksheetName = request.args.get('worksheetName')
    periods = request.args.get('periods')
    start = request.args.get('start')
    end = request.args.get('end')
    
    scope_app =['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive'] 
    cred = ServiceAccountCredentials.from_json_keyfile_name('token.json',scope_app) 
    client = gspread.authorize(cred)
    sheet = client.open(sheetName)
    worksheet = sheet.worksheet(worksheetName)
    data = worksheet.get_all_values()
    return dataProcessing(data, int(periods), start, end)

if __name__ == '__main__':
    app.run(debug=True, port=5000)