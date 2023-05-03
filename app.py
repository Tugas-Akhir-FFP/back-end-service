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
    s= max(results, key=lambda x: x[-1])
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
    index = pd.date_range('01-01-2015', '21-12-2022', freq='D')
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df[:-10]
    df  = df.rename(columns={'Tanggal':'Date','Tavg':'Temperature','RH_avg':'Humidity','ff_avg':'Wind','RR':'Rainfall'})
    
    print(df['Date'])
    
    df = df.drop(df.index[0]) 
    df = df.drop(df.columns[0], axis=1)
    # start = df['Tanggal'][1]
    # index = pd.date_range(df['Date'], df['Date'][-1:], freq='D')
    # print(df['Date'])
    # df = df.set_index(index)
    # df = df.replace('',math.nan)
    # df = df.replace('8888',math.nan)
    # df = df.astype('float32')
    # df = df.fillna(df.mean())
    # df = df.loc[start:end]
    # df = df.reset_index(drop=True)
    # Tanggal = pd.date_range('2019-12-31', periods=periods, freq='D')
    # Tanggal = Tanggal.tolist()
    response = {}
    # print(df['Rainfall'])
    response['Hasil'] = df['Temperature'].tolist()
    # response['Tanggal'] = Tanggal
    # for i in range(3):
    #     response[df.columns[i]] = grid_search(df[df.columns[i]])
    return response
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
    app.run(debug=True, host='127.0.0.1', port=5000)