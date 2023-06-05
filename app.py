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
import skfuzzy as fuzz 
from skfuzzy import control as ctrl
import decimal
from statsmodels.tsa.holtwinters import ExponentialSmoothing 

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
    print(len(df),"panjang data")
    #Kotawaringin df[1:1451], df[1451:1471]
    #Sidoarjo df[1:2439], df[2439:2459]wewewew
    train, test = df[1:1451], df[1451:1471]
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

def Prediction(df, seasonal, trend, periods, slevel, stren, sseasonal, start, end):
    print(len(df))
    # low = 1451
    # high = 1471
    low = df.index.get_loc(start)
    high = df.index.get_loc(end)
    print(low, high)
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

    return {
        'predictions' : predictions.tolist(),
        'mse' : mse,
        'r2' : r2,
        'rmse' : rmse
    }
#create function for formula fwi calculation using 4 parameter
def fwiCalculation(temp, rh, wind, rainfall, month =1):
    
    dmc_prev = 6
    dc_prev = 15
    FFMC_prev = 85

    m_prev = 147.2 * (101.0 - FFMC_prev) / (59.5 + FFMC_prev)

    # Calculate mo
    if rainfall > 0.5:
        rf = rainfall - 0.5
        if m_prev > 150.0:
            mo = m_prev + (42.5 * rf * math.exp(-100.0 / (251.0 - m_prev)) * (1.0 - math.exp(-6.93 / rf))) + (0.0015 * (m_prev - 150.0) ** 2) * math.sqrt(rf)
        else:
            mo = m_prev + 42.5 * rf * math.exp(-100.0 / (251.0 - m_prev)) * (1.0 - math.exp(-6.93 / rf))
    else:
        mo = m_prev

    # Calculate Ed
    Ed = 0.942 * (rh ** 0.679) + (11.0 * math.exp((rh - 100.0) / 10.0)) + 0.18 * (21.1 - temp) * (1.0 - 1.0 / math.exp(0.115 * rh))

    # Calculate m
    if Ed > mo:
        Ew = 0.618 * (rh ** 0.753) + (10.0 * math.exp((rh - 100.0) / 10.0)) + 0.18 * (21.1 - temp) * (1.0 - 1.0 / math.exp(0.115 * rh))
        if mo < Ew:
            k1 = 0.424 * (1.0 - (1.0 - rh / 100.0) ** 1.7) + (0.0694 * math.sqrt(wind)) * (1.0 - (1.0 - rh / 100.0) ** 8)
            kw = k1 * (0.581 * math.exp(0.0365 * temp))
            m = Ew - (Ew - mo) * math.exp(-kw)
        else:
            m = mo
    else:
        k0 = 0.424 * (1.0 - ((rh / 100.0) ** 1.7)) + (0.0694 * math.sqrt(wind)) * ((rh / 100.0) ** 8)
        kd = k0 * (0.581 * math.exp(0.0365 * temp))
        m = mo + (Ed - mo) * math.exp(-kd)

    # Calculate FFMC
    FFMC = (59.5 * (250.0 - m)) / (147.2 + m)
    print(FFMC, "FFMC")
    
    # Constants
    el = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    t = 0.5 * el[month-1]
    
    # Calculate mth
    if temp < -1.1:
        mth = 0
    else:
        mth = 1.894 * (temp + 1.1) * (100 - rh) * 12 * 10 ** (-6)
    # print(mth,"mth")
    # Calculate dmc with rainfall effect
    if rainfall > 1.5:
        ra = rainfall
        rw = 0.92 * ra - 1.27
        wmi = 20.0 + math.exp(5.6348 - 100.0 / (temp + 3.43))
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        pr = 244.72-43.43 * math.log(wmr - 20.0)
        # print(pr, "wmr")
        
    else:
        pr = dmc_prev

    if pr < 0.0:
        pr = 0.0
    
    # Calculate dmc without rainfall effect
    dmc = pr + 100.0 * mth
    print(dmc, "DMC")

    if rainfall > 2.8:
        Pd = 0.83 * rainfall - 1.27
        
        ## Calculate Qprev
        Qprev = 800 * math.exp(-dc_prev / 400)

        ## Calculate Qrt
        Qrt = Qprev + 3.937 * Pd

        ## Calculate DCrt
        Dcrt = 400 * math.log(800 / Qrt)

        if Dcrt < 0:
            Dcrt = 0
    else:
        Dcrt = dc_prev

    ## Calculate V
    Lf = 12
    if temp < -2.8:
        temp = -2.8
    V = 0.36 * (temp + 2.8) + Lf

    ## Calculate DC
    DC = Dcrt + 0.5 * V
    print(DC,"DC")
    
    
    if dmc <= 0.4 * DC:
        bui = 0.8 * (dmc * DC) / (dmc + 0.4 * DC)
    else:
        bui = dmc - (1 - (0.8 * DC / (dmc + 0.4 * DC))) * (0.92 + (0.0114 * dmc) ** 1.7)
    
    # ISI calculate
    m = 147.2 * (101.0 - FFMC) / (59.5 + FFMC)
    Fu = math.exp(0.05039 * wind*3.6)
    Ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + (m ** 5.31) / (4.93 * (10 ** 7)))
    isi = Ff * Fu* 0.208
    print(isi, "ISI")
    # calculate FWI
    
    if bui <= 80.0:
        bb = 0.1 * isi * (0.626 * (bui ** 0.809) + 2.0)
    else:
        bb = 0.1 * isi * (1000.0 / (25 + 108.64 *math.exp(-0.023 * bui)))
    if bb <= 1.0:
        fwi = bb
    else:
        fwi = math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)
    #Return with round 2 decimalsss
    return round(fwi,2)

def calculate_fwi_list(temperature_list, humidity_list, wind_list, rainfall_list):
    fwi_list = []
    for i in range(len(temperature_list)):
        fwi = fwiCalculation(temperature_list[i], humidity_list[i], wind_list[i], rainfall_list[i])
        fwi_list.append(fwi)
    return fwi_list

def dataProcessing(data, periods, start, end, freq='D'):
    def pre_Fix_data(data) :
        data = data.replace(['8888', ''], np.nan)
        data = data.astype(float)
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data
    
    index = pd.date_range(start, end, freq=freq)
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df[:-periods]
    df = df.rename(columns={'Tanggal':'Date','Tx':'Temperature','RH_avg':'Humidity','ff_avg':'Wind','RR':'Rainfall'})
    
    #filter data by range date start and end
    df = df.drop(df.index[0]) 
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # df_filtered = df[df['Date'].between(start, end)]
    
    
    parameters = []
    parameters_names = ['Temperature', 'Humidity', 'Wind', 'Rainfall']

    for param_name in parameters_names : 
        param_data = df[['Date', param_name]].set_index('Date')
        param_data = pre_Fix_data(param_data).resample('D').mean()
        param_list = param_data[param_name].tolist()

        parameters.append({
            'name': param_name,
            'data': param_data,
            'trend': None if param_name in ['Temperature', 'Rainfall'] else 'multiplicative' if param_name == 'Humidity' else 'additive',
            'seasonal': None
        })

        globals()[f'{param_name.lower()}_list'] = param_list

    # date_list = df_filtered['Date'].tolist()

    

    predict_result = []
    error_result = []
    # date list from start to end
    date_list = pd.date_range(start, end, freq=freq).tolist()
    # Looping for prediction
    for param in parameters:
        result = Prediction(param['data'], param['seasonal'], param['trend'], 4, 0.9, 0.1, 0.1, start, end)
        predict_result.append({param['name']: result['predictions']})

        error_result.append({

            param['name']: {
                'MSE': result['mse'],
                'RMSE': result['rmse'],
                'R2': result['r2']
            }
    })



    ## Fuzzy Universe
    def fuzzy(value):
        result=[]
        fwi = ctrl.Antecedent(np.arange(0, 20, 1), 'x') 
        fwi['biru'] = fuzz.trapmf(fwi.universe, [0, 0, 1, 2])
        fwi['hijau'] = fuzz.trapmf(fwi.universe, [1, 2, 6, 7])
        fwi['kuning'] = fuzz.trimf(fwi.universe, [6, 7, 13])
        fwi['merah'] = fuzz.trapmf(fwi.universe, [7, 13, np.inf, np.inf])

        fwi_level_biru = fuzz.interp_membership(fwi.universe, fwi['biru'].mf, value)
        fwi_level_hijau = fuzz.interp_membership(fwi.universe, fwi['hijau'].mf, value)
        fwi_level_kuning = fuzz.interp_membership(fwi.universe, fwi['kuning'].mf, value)
        fwi_level_merah = fuzz.interp_membership(fwi.universe, fwi['merah'].mf, value)
        result = [fwi_level_biru, fwi_level_hijau, fwi_level_kuning, fwi_level_merah]
        plt.plot(fwi.universe, fwi['biru'].mf, 'b', linewidth=1.5, label='Biru')
        plt.plot(fwi.universe, fwi['hijau'].mf, 'g', linewidth=1.5, label='Hijau')
        plt.plot(fwi.universe, fwi['kuning'].mf, 'y', linewidth=1.5, label='Kuning')
        plt.plot(fwi.universe, fwi['merah'].mf, 'r', linewidth=1.5, label='Merah')
        # plt.show()
        return result

    # Get Prediction result from each parameter list 
    temperature_prediction = predict_result[0]['Temperature']
    humidity_prediction = predict_result[1]['Humidity']
    wind_prediction = predict_result[2]['Wind']
    rainfall_prediction = predict_result[3]['Rainfall']

    #implement to calculate fwi from all parameters
    fwi_values = calculate_fwi_list(temperature_prediction, humidity_prediction, wind_prediction, rainfall_prediction)
    fuzzy_result = []
    for i in range(len(fwi_values)): 
        result = fuzzy(fwi_values[i])
        fuzzy_result.append({'FWI Value' : fwi_values[i], 'Fuzzy' : result})
    

    data_result = []
    #looping to get data from each parameter
    for i in range(len(predict_result[0]['Temperature'])):
        result = {}
        for param in parameters:
            param_name = param['name']
            param_predictions = predict_result[parameters.index(param)][param_name]
            result[param_name] = {
                'Data': date_list[i],
                'Prediction': param_predictions[i],
                'Fuzzy': {
                    'FWI Value': fwi_values[i],
                    'Fuzzy': fuzzy_result[i]['Fuzzy']
                }
            }
        data_result.append(result)

    response = {
        'Error Result' : error_result,
        'Data Result' : data_result,
    }
    # for i in range(len(date_list)):
    #     data = {
            
    #         'Date': date_list[i],
    #         'Temperature': temperature_list[i],
    #         'Humidity': humidity_list[i],
    #         'Wind': wind_list[i],
    #         'Rainfall': rainfall_list[i],
    #     }
        
    #     response['Parameter'].append(data)

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
    app.run(debug=True, port=3000)