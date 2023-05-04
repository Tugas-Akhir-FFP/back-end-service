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
    train, test = df[1:200], df[200:210]
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
                                print('error')
    
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
    #change prediction value round to 2 decimal
    predictions = np.array(predictions).flatten().round(2)
    test = np.array(test).flatten()

    #print type of data test
    print(type(test))
    print(type(predictions))
    r2 = r2_score(test, predictions)
    # mse = mean_squared_error(test, predictions)  
    mse = np.square(np.subtract(test,predictions)).mean()
    rmse = math.sqrt(mse)
    print(np.array(test))
    hasil = {}
    hasil['Hasil'] = predictions.tolist()   
    print(r2,'r2')
    print(mse,'mse')
    print(rmse,'rmse')
    print(test)
    print(best_params)
    return predictions.tolist() 

#create function for formula fwi calculation using 4 parameter
def fwiCalculation(temperature, humidity, wind, rainfall):
    #calculate ffmc
    ffmc = 0.0 
    if temperature > -1.1 and temperature < 30.0 :
        ffmc = (59.5 * (math.exp((temperature - 10.0) / -6.0))) - (14.0 * humidity) + (0.5 * wind) + 43.5
        if ffmc < 0.0 :
            ffmc = 0.0
    elif temperature >= 30.0 :
        ffmc = (122.0 * (math.exp((temperature - 10.0) / -6.0))) - (0.2 * (100.0 - humidity)) + (wind * 0.1) + 50.0
        if ffmc > 101.0 :
            ffmc = 101.0

    #calculate DC
    dc = (ffmc - 30.0) * 0.5 + 3.0 * (wind / 20.0)

    #calculate ISI
    isi = 0.0 
    if wind > 40.0 :
        isi = 16.0
    else :
        isi = (wind / 4.0) * (math.exp(0.05039 * humidity)) * 0.01

    #calculate BUI
    bui = 0.0
    if dc <= 0.0 :
        bui = 0.0
    else :
        bui = (dc / 10.0) * (0.5 + 0.3 * math.log10(rainfall + 1.0))

    #calculate FWI
    fwi = 0.0
    if bui <= 80.0 :
        fwi = bui * 0.1 + isi * 0.4
    else : 
        fwi = bui * 0.6 + isi * 0.4

    return fwi

def calculate_fwi_list(temperature_list, humidity_list, wind_list, rainfall_list):
    fwi_list = []
    for i in range(len(temperature_list)):
        fwi = fwiCalculation(temperature_list[i], humidity_list[i], wind_list[i], rainfall_list[i])
        fwi_list.append(fwi)
    return fwi_list

def dataProcessing(data, periods, start, end, freq='D'): 
    index = pd.date_range(start, end, freq=freq)
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df[:-periods]
    df = df.rename(columns={'Tanggal':'Date','Tavg':'Temperature','RH_avg':'Humidity','ff_avg':'Wind','RR':'Rainfall'})
    
    #filter data by range date start and end
    df = df.drop(df.index[0]) 
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df_filtered = df[df['Date'].between(start, end)]
    
    
    # select temperature data
    temperature_data = df_filtered[['Date', 'Temperature']]
    temperature_data = temperature_data.set_index('Date')
    temperature_list = temperature_data['Temperature'].tolist()
    date_list = df_filtered['Date'].tolist()

    #Replace 8888 with nan, null value with nan, fill nan with previous value, fill nan with next value for temperature
    temperature_data = temperature_data.replace('8888', np.nan)
    temperature_data = temperature_data.replace('', np.nan)
    temperature_data = temperature_data.fillna(method='ffill')
    temperature_data = temperature_data.fillna(method='bfill')
    temperature_data = temperature_data.astype(float)
    temperature_data = temperature_data.reset_index()
    temperature_data = temperature_data.set_index('Date')
    temperature_data = temperature_data.resample('D').mean()
    #return to list
    temperature_list = temperature_data['Temperature'].tolist()

    # Select humidity data
    humidity_data = df_filtered[['Date', 'Humidity']]
    humidity_data = humidity_data.set_index('Date')
    humidity_list = humidity_data['Humidity'].tolist()
    date_list = df_filtered['Date'].tolist()

    #Replace 8888 with nan, null value with nan, fill nan with previous value, fill nan with next value for humidity
    humidity_data = humidity_data.replace('8888', np.nan)
    humidity_data = humidity_data.replace('', np.nan)
    humidity_data = humidity_data.fillna(method='ffill')
    humidity_data = humidity_data.fillna(method='bfill')
    humidity_data = humidity_data.astype(float)
    humidity_data = humidity_data.reset_index()
    humidity_data = humidity_data.set_index('Date')
    humidity_data = humidity_data.resample('D').mean()
    #return to list
    humidity_list = humidity_data['Humidity'].tolist()

    # Select wind data
    wind_data = df_filtered[['Date', 'Wind']]
    wind_data = wind_data.set_index('Date')
    wind_list = wind_data['Wind'].tolist()
    date_list = df_filtered['Date'].tolist()

    #Replace 8888 with nan, null value with nan, fill nan with previous value, fill nan with next value for wind
    wind_data = wind_data.replace('8888', np.nan)
    wind_data = wind_data.replace('', np.nan)
    wind_data = wind_data.fillna(method='ffill')
    wind_data = wind_data.fillna(method='bfill')
    wind_data = wind_data.astype(float)
    wind_data = wind_data.reset_index()
    wind_data = wind_data.set_index('Date')
    wind_data = wind_data.resample('D').mean()
    #return to list
    wind_list = wind_data['Wind'].tolist()


    # Select rainfall data
    rainfall_data = df_filtered[['Date', 'Rainfall']]
    rainfall_data = rainfall_data.set_index('Date')
    rainfall_list = rainfall_data['Rainfall'].tolist()
    date_list = df_filtered['Date'].tolist()

    #Replace 8888 with nan, null value with nan, fill nan with previous value, fill nan with next value for rainfall
    rainfall_data = rainfall_data.replace('8888', np.nan)
    rainfall_data = rainfall_data.replace('', np.nan)
    rainfall_data = rainfall_data.fillna(method='ffill')
    rainfall_data = rainfall_data.fillna(method='bfill')
    rainfall_data = rainfall_data.astype(float)
    rainfall_data = rainfall_data.reset_index()
    rainfall_data = rainfall_data.set_index('Date')
    rainfall_data = rainfall_data.resample('D').mean()
    #return to list
    rainfall_list = rainfall_data['Rainfall'].tolist()

    #Create new variable to implement grid search for all parameters
    grid_temp = grid_search(temperature_data)
    grid_humidity = grid_search(humidity_data)
    grid_wind = grid_search(wind_data)
    grid_rainfall = grid_search(rainfall_data)

    #implement to calculate fwi from all parameters
    fwi_values = calculate_fwi_list(grid_temp, grid_humidity, grid_wind, grid_rainfall)
    
    
    #hasil semua prediksi parameter
    print("--------  TEMPERATURE  --------")
    print(grid_temp)
    print("--------  HUMIDITY  --------")
    print(grid_humidity)
    print("--------  WIND  --------")
    print(grid_wind)
    print("--------  RAINFALL  --------")
    print(grid_rainfall)

    #hasil prediksi fwi
    print("--------  FWI  --------")
    print(fwi_values)


    response = {
        'Date' : date_list,
        'Temperature' : temperature_list,
        'Humidity' : humidity_list,
        'Wind' : wind_list,
        'Rainfall' : rainfall_list
    }

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
    app.run(debug=True, port=5000)