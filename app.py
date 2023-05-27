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
import skfuzzy as fuzz 
from skfuzzy import control as ctrl

from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from joblib import Parallel, delayed

app = Flask(__name__)
api = Api(app)
CORS(app)

param_grid = {'seasonal': ['additive', 'multiplicative', None],
            'trend': ['additive', 'multiplicative', None],
            'seasonal_periods': [4, 12],
            'smoothing_level': np.linspace(0.1, 0.9, 9),
            'smoothing_trend': np.linspace(0.1, 0.9, 9),
            'smoothing_seasonal': np.linspace(0.1, 0.9, 9)}

def grid_search(df):
    print(len(df))
    #Kotawaringin df[1:1451], df[1451:1471]
    #Sidoarjo df[1:2439], df[2439:2459]
    train, test = df[1:1471], df[1451:1471]
    print(len(train))
    print(len(test))

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
                                # predictions = model_fit.predict(start=len(train), end = len(train)+len(test)-1)

                                #create predict 20 data
                                predictions = model_fit.predict(start=1451, end= 1471-1)
                                r2 = r2_score(test, predictions)
                                mse = mean_squared_error(test, predictions)
                                rmse = math.sqrt(mse)
                                results.append((seasonal, trend, seasonal_period, smoothing_level, smoothing_trend, smoothing_seasonal,r2))
                            except:
                                print('error')
    
    # Cari parameter yang menghasilkan r-square terbaik, udah di save
    best_params = max(results,key=lambda x: x[-1])
    model = ExponentialSmoothing(train,
                                 seasonal=best_params[0],
                                 trend=best_params[1],
                                 seasonal_periods=best_params[2])
    model_fit = model.fit(smoothing_level=best_params[3], 
                          smoothing_trend=best_params[4], 
                          smoothing_seasonal=best_params[5])
    predictions = model_fit.predict(start=1451, end= 1471-1)
    #change prediction value round to 2 decimal
    predictions = np.array(predictions).flatten().astype(int)
    test = np.array(test).flatten().astype(int)

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
    print("-------- Kalkulasi hasil Grid------------") 
    print(r2,'r2')
    print(mse,'mse')
    print(rmse,'rmse')

    print("========== HASIL UNTUK PARAMETER TERBAIK ==========")
    print(best_params,"best param")
    print("HASIL Prediksi : " ,predictions)
    print("Data Test : " ,test)
    return predictions.tolist() 

# def new_gridSC(data):
#     # Define Grid Search Parameter
#     param_grid = {
#         'trend': ['add', 'mul', None],
#         'seasonal': ['add', 'mul', None],
#         'seasonal_periods': [4, 12],
#         'smoothing_level': np.linspace(0.1, 0.9, 9),
#         'smoothing_trend': np.linspace(0.1, 0.9, 9),
#         'smoothing_seasonal': np.linspace(0.1, 0.9, 9)
#     }
    
#     train, test = data.iloc[1:2439], data.iloc[2439:2459]

#     def fit_model(params, data):
#         model = ExponentialSmoothing(data, trend=params['trend'], seasonal=params['seasonal'],
#                                      seasonal_periods=params['seasonal_periods'],
#                                      smoothing_level=params['smoothing_level'],
#                                      smoothing_trend=params['smoothing_trend'],
#                                      smoothing_seasonal=params['smoothing_seasonal'])
#         score = model.fit().aic
#         return score
    
#     grid_search = GridSearchCV(estimator=ExponentialSmoothing(endog=train), param_grid=param_grid)
#     results = Parallel(n_jobs=-1)(
#         delayed(fit_model)(params, train) for params in grid_search.param_grid
#     )

#     # Find best parameter combination
#     best_index = np.argmin(results)
#     best_params = grid_search.param_grid[best_index]

#     print("Best parameter:", best_params)
#     print("Best AIC Score:", results[best_index])

#     # Fit the best model to training data
#     best_model = ExponentialSmoothing(train, trend=best_params['trend'], seasonal=best_params['seasonal'],
#                                       seasonal_periods=best_params['seasonal_periods'],
#                                       smoothing_level=best_params['smoothing_level'],
#                                       smoothing_trend=best_params['smoothing_trend'],
#                                       smoothing_seasonal=best_params['smoothing_seasonal'])
#     best_model.fit()

#     # Evaluate the best model
#     Prediction = best_model.predict(start=len(train), end=len(train) + len(test) - 1)

#     return best_model, Prediction        



def Prediction(df, seasonal, trend, periods, slevel, stren, sseasonal):
    print(len(df))
    low = 1312
    high = 1332

    #kota waringin barat
    # 8 Agustus 2022 Low = 1312 High = 1332
    # 1 januari 2023 Low = 1451 High = 1471
    # 17 Mei 2023 Low = 1580 High = 1599

    # kab Sidoarjo
    # 31 Maret 2023 Low = 3002 High = 3012
    # low = 1312
   # high = 1332

    # Kab Gresik 
    # low = 1312
   # high = 1332
   
    train, test = df[1:high], df[low:high]
    print(len(train), len(test))
    model = ExponentialSmoothing(train,
                                seasonal=seasonal,
                                trend=trend,
                                seasonal_periods=periods)
    model_fit = model.fit(smoothing_level=slevel, 
                        smoothing_trend=stren, 
                        smoothing_seasonal=sseasonal)
    
    predictions = model_fit.predict(start=low, end= high-1)
    predictions = np.array(predictions).flatten().astype(int)
    test = np.array(test).flatten().astype(int)
    mse = np.square(np.subtract(test,predictions)).mean()
    r2 = r2_score(test, predictions)
    rmse = math.sqrt(mse)
    print("------------- PERHITUNGAN ERROR--------")
    print("Data Test : ", test)
    print("Prediksi : ", predictions)
    print("Mse : ", mse)
    print("R2 : ", r2)
    print("RMSE : ", rmse)
    # #error calculation
    # mse = np.square(np.subtract(test,predictions)).mean()
    # rmse = math.sqrt(mse)
    # r2 = r2_score(test, predictions)
    # print(mse,'mse')
    # print(rmse,'rmse')
    # print(r2,'r2')

    return predictions.tolist()
#create function for formula fwi calculation using 4 parameter
def fwiCalculation(temperature, humidity, wind, rainfall):
    # #calculate ffmc
    # ffmc = 0.0 
    # if temperature > -1.1 and temperature < 30.0 :
    #     ffmc = (59.5 * (math.exp((temperature - 10.0) / -6.0))) - (14.0 * humidity) + (0.5 * wind) + 43.5
    #     if ffmc < 0.0 :
    #         ffmc = 0.0
    # elif temperature >= 30.0 :
    #     ffmc = (122.0 * (math.exp((temperature - 10.0) / -6.0))) - (0.2 * (100.0 - humidity)) + (wind * 0.1) + 50.0
    #     if ffmc > 101.0 :
    #         ffmc = 101.0

    # #calculate DC
    # dc = (ffmc - 30.0) * 0.5 + 3.0 * (wind / 20.0)

    # #calculate ISI
    # isi = 0.0 
    # if wind > 40.0 :
    #     isi = 16.0
    # else :
    #     isi = (wind / 4.0) * (math.exp(0.05039 * humidity)) * 0.01

    # #calculate BUI
    # bui = 0.0
    # if dc <= 0.0 :
    #     bui = 0.0
    # else :
    #     bui = (dc / 10.0) * (0.5 + 0.3 * math.log10(rainfall + 1.0))

    # #calculate FWI
    # fwi = 0.0
    # if bui <= 80.0 :
    #     fwi = bui * 0.1 + isi * 0.4
    # else : 
    #     fwi = bui * 0.6 + isi * 0.4

    # m1 = 2.303 * (math.log10(humidity) - 0.4343 * math.log10(100 - humidity))
    # m2 = 0.987 * (math.log10(rainfall + 0.1))
    # m3 = 0.0345 * temperature
    # m4 = 1.59 * (math.log10(wind + 0.396))
    # fwi = m1 + m2 + m3 + m4

    # try:
    #     m1 = 2.303 * (math.log10(humidity) - 0.4343 * math.log10(100 - humidity))
    #     m2 = 0.987 * (math.log10(rainfall + 0.1))
    #     m3 = 0.0345 * temperature
    #     m4 = 1.59 * (math.log10(wind + 0.396))
    #     fwi = m1 + m2 + m3 + m4
    #     return fwi
    # except (ValueError, ZeroDivisionError) as e:
    #     print("Invalid input values:", e)
    #     return None
    FFMC_old = 85.0
    DMC_old = 6.0
    DC_old = 15.0

    # Calculate FFMC
    FFMC_new = (FFMC_old + 0.0278 * DMC_old * math.exp(0.0385 * (temperature - 20.0))) * (1.0 - math.exp(-0.1 * humidity))

    # Calculate DMC
    DMC_new = (DMC_old + 0.1 * rainfall) * math.exp(0.1 * (temperature - 20.0))

    # Calculate DC
    DC_new = (DC_old + 1.5 * (rainfall - 1.5)) * math.exp(0.023 * (temperature - 20.0))

    # Calculate ISI
    ISI = 0.4 * wind

    # Calculate BUI
    BUI = 0.5 * (DMC_new + DC_new) / (10.0 - 0.1 * rainfall)

    # Calculate FWI
    FWI = (ISI + BUI) / 2.0
    return FWI

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
    temperature_data = temperature_data.replace('0',np.nan)
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
    humidity_data = humidity_data.replace('0',np.nan)
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
    wind_data = wind_data.replace('0',np.nan)
    wind_data = wind_data.fillna(method='ffill')
    wind_data = wind_data.fillna(method='bfill')
    wind_data = wind_data.astype(float)
    wind_data = wind_data.reset_index()
    wind_data = wind_data.set_index('Date')
    wind_data = wind_data.resample('D').median()
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
    rainfall_data = rainfall_data.replace('0',np.nan)
    rainfall_data = rainfall_data.fillna(method='ffill')
    rainfall_data = rainfall_data.fillna(method='bfill')
    rainfall_data = rainfall_data.astype(float)
    rainfall_data = rainfall_data.reset_index()
    rainfall_data = rainfall_data.set_index('Date')
    # rainfall_data = rainfall_data.resample('D').mean()
    rainfall_data = rainfall_data.resample('D').median()
    #return to list
    rainfall_list = rainfall_data['Rainfall'].tolist()

    #Create new variable to implement grid search for all parameters
    # grid_temp = grid_search(temperature_data)
    # grid_humidity = grid_search(humidity_data)
    # grid_wind = grid_search(wind_data)
    # grid_rainfall = grid_search(rainfall_data)

    # ## New Grid Search
    # temp_grid = new_gridSC(temperature_data)
    # humidity_grid = new_gridSC(humidity_data)
    # wind_grid = new_gridSC(wind_data)
    # rainfall_grid = new_gridSC(rainfall_data)


    # print("----- HASIL NEW GRID SEARCH -----")
    # print(temp_grid)
    # print(humidity_grid)
    # print(wind_grid)
    # print(rainfall_grid)

    

    # temp = Prediction(temperature_data,'additive', 'multiplicative', 12, 0.5, 0.1, 0.2)
    # humidity = Prediction(humidity_data,'additive', 'additive', 12, 0.1, 0.9, 0.5)
    # wind = Prediction(wind_data,'multiplicative', 'additive', 12, 0.9, 0.1,0.2)
    # rainfall = Prediction(rainfall_data,'additive', 'additive', 4, 0.9, 0.4,0.2 )

    # # Kabupaten Kotawaringin Barat
    # temp = Prediction(temperature_data,'additive', 'multiplicative', 12, 0.6, 0.4, 0.2)
    # humidity = Prediction(humidity_data,'multiplicative', 'additive', 12, 0.1, 0.5, 0.1)
    # wind = Prediction(wind_data,'multiplicative', 'multiplicative', 12, 0.1, 0.7,0.1)
    # rainfall = Prediction(rainfall_data,'additive', 'additive', 12, 0.1, 0.1,0.1)

    # Kabupaten Kotawaringin Barat versi baru
    temp = Prediction(temperature_data,None, None, 4, 0.9, 0.1, 0.1)
    humidity = Prediction(humidity_data,None, 'multiplicative', 4, 0.9, 0.1, 0.1)
    wind = Prediction(wind_data,None, 'additive', 4, 0.9, 0.1,0.1)
    rainfall = Prediction(rainfall_data,None, None, 4, 0.9, 0.1,0.1)

    # ## Kabupaten Sidoarjo
    # temp = Prediction(temperature_data,'multiplicative', 'multiplicative', 4, 0.5, 0.3, 0.6)
    # humidity = Prediction(humidity_data,'multiplicative', 'additive', 12, 0.1, 0.5, 0.1)
    # wind = Prediction(wind_data,'multiplicative', 'multiplicative', 12, 0.1, 0.7,0.1)
    # rainfall = Prediction(rainfall_data,'additive', 'additive', 12, 0.1, 0.1,0.1)
    
    # grid_result = [
    #     {'Temperature' : grid_temp},
    #     {'Humidity' : grid_humidity},
    #     {'Wind' : grid_wind},
    #     {'Rainfall' : grid_rainfall},
    # ]

    # print("----- HASIL GRID SEARCH -----")
    # print(grid_temp)
    # print(grid_humidity)
    # print(grid_wind)
    # print(grid_rainfall)

    predict_result = [
        {'Temperature' : temp},
        {'Humidity' : humidity},
        {'Wind' : wind},
        {'Rainfall' : rainfall}
    ]


    #Fuzzy Universe
    def fuzzy(value):
        result=[]
        fwi = ctrl.Antecedent(np.arange(0, 20, 1), 'x') # type: ignore
        fwi['biru'] = fuzz.trapmf(fwi.universe, [0, 0, 1, 2])
        fwi['hijau'] = fuzz.trapmf(fwi.universe, [1, 2, 6, 7])
        fwi['kuning'] = fuzz.trapmf(fwi.universe, [6, 7, 13,13])
        fwi['merah'] = fuzz.trapmf(fwi.universe, [7,13,13,13])

        fwi_level_biru = fuzz.interp_membership(fwi.universe, fwi['biru'].mf, value)
        fwi_level_hijau = fuzz.interp_membership(fwi.universe, fwi['hijau'].mf, value)
        fwi_level_kuning = fuzz.interp_membership(fwi.universe, fwi['kuning'].mf, value)
        fwi_level_merah = fuzz.interp_membership(fwi.universe, fwi['merah'].mf, value)
        result = [fwi_level_biru, fwi_level_hijau, fwi_level_kuning, fwi_level_merah]
        return result

    #implement to calculate fwi from all parameters
    fwi_values = calculate_fwi_list(temp, humidity, wind, rainfall)
    fuzzy_result = []
    for i in range(len(fwi_values)): 
        result = fuzzy(fwi_values[i])
        fuzzy_result.append({'Data' : fwi_values[i], 'Fuzzy' : result})
    
    response = {
        # 'Grid Search' : grid_result,
        'Prediction' : predict_result,
        'Fuzzy Result' : fuzzy_result,
        'Parameter' : [],
        'Fwi Result' : fwi_values
    }
    for i in range(len(date_list)):
        data = {
            
            'Date': date_list[i],
            'Temperature': temperature_list[i],
            'Humidity': humidity_list[i],
            'Wind': wind_list[i],
            'Rainfall': rainfall_list[i],
        }
        
        response['Parameter'].append(data)

    # json_data = json.dumps(response, indent=4)
    return response



@app.route('/api')
def get_credentials():
    sheetName = request.args.get('sheetName')
    worksheetName = request.args.get('worksheetName')
    periods = request.args.get('periods')
    start = request.args.get('start')
    end = request.args.get('end')
    
    scope_app =['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive'] 
    cred = ServiceAccountCredentials.from_json_keyfile_name('token.json',scope_app)  # type: ignore
    client = gspread.authorize(cred)
    sheet = client.open(sheetName)
    worksheet = sheet.worksheet(worksheetName)
    data = worksheet.get_all_values()

    return dataProcessing(data, int(periods), start, end)# type: ignore

if __name__ == '__main__':
    app.run(debug=True, port=5000)