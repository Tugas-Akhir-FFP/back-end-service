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
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler

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

    #Kotawaringin df[1:1451], df[1451:1471]
    # #Sidoarjo df[1:2439], df[2439:2459]wewewew
    # train, test = df[1:high], df[low:high]
    mid = 1613
    train = df[0:mid]
    actual = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 10.6, 0.0, 0.0, 0.0, 0.0, 10.0, 9.3, 0.8, 6.5, 25.8, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 20.0, 0.0, 16.7, 15.7, 15.7, 16.0, 16.7, 15.7, 15.0])

    # data = train.values
    # x = z_score(data)
    x = train
    fore = 30
    results = []
    for seasonal in param_grid['seasonal']:
        for trend in param_grid['trend']:
            for seasonal_period in param_grid['seasonal_periods']:
                for smoothing_level in param_grid['smoothing_level']:
                    for smoothing_trend in param_grid['smoothing_trend']:
                        for smoothing_seasonal in param_grid['smoothing_seasonal']:
                            try:
                                model = ExponentialSmoothing(x, 
                                                            seasonal=seasonal, 
                                                            trend=trend, 
                                                            seasonal_periods=seasonal_period)
                                model_fit = model.fit(smoothing_level=smoothing_level, 
                                                    smoothing_trend=smoothing_trend, 
                                                    smoothing_seasonal=smoothing_seasonal)
                                # Make predictions and calculate R-squared
                                # predictions = model_fit.predict(start=len(train), end = len(train)+len(test)-1)

                                #create predict 20 data
                                #predict sama as data test
                                predictions = model_fit.forecast(steps=fore)
                                # predictions = z_score_DeStandardization(predictions, data)
                                predictions = np.array(predictions).flatten().__abs__().round(1)
                                
                                n = len(actual)
                                r = (n*(sum(actual*predictions)) - sum(actual)*sum(predictions)) / math.sqrt((n*sum(actual**2) - sum(actual)**2)*(n*sum(predictions**2) - sum(predictions)**2))

                                mse = np.square(np.subtract(actual,predictions)).mean()
                                r2 = r**2
                                r2 = round(r2,2)                              
                                # test = z_score(test)
                                # predictions = z_score(predictions)
                                print(predictions, "Ini hasil forecastnya bwang : ")
                                # r2 = r2_score(actual, predictions)
                                
                                results.append((seasonal,trend, seasonal_period, smoothing_level, smoothing_trend, smoothing_seasonal,mse))
                            except:
                                print('error')
    
    # Cari parameter yang menghasilkan r-square terbaik, udah di save
    best_params = min(results,key=lambda x: x[-1])
    model = ExponentialSmoothing(train,
                                seasonal=best_params[0],
                                trend=best_params[1],
                                seasonal_periods=best_params[2])
    model_fit = model.fit(smoothing_level=best_params[3], 
                          smoothing_trend=best_params[4], 
                          smoothing_seasonal=best_params[5])
    predictions = model_fit.forecast(steps=fore)
    #change prediction value round to 2 decimal
    predictions = np.array(predictions).flatten().__abs__().round(1)
    test = np.array(actual).flatten()

    mse = np.square(np.subtract(actual,predictions)).mean()
    r2 = r2_score(actual, predictions)
    rmse = math.sqrt(mse)
    # mape = np.mean((np.abs(predictions - test)/np.abs(test))* 100)
    hasil = {}
    hasil['Hasil'] = predictions.tolist()  
    print("-------- Kalkulasi hasil Grid Forecast------------") 
    print(r2,'r2')
    print(mse,'mse')
    print(rmse,'rmse')
    # print(mape,'mape')

    print("========== HASIL UNTUK PARAMETER TERBAIK ==========")
    print(best_params,"best param")
    print("HASIL Prediksi : " ,best_params)
    print("Data Test : " ,test)
    return predictions.tolist() 

# Z-Score Standardization
def z_score_calculation(data):
    # Create a copy of the data to avoid modifying the original array
    data = np.copy(data)

    # Create conditional for same value
    unique_values, value_counts = np.unique(data, return_counts=True)
    if np.all(value_counts > 1):
        # Add 0.1 to every other occurrence of the duplicate values
        add_value = 0.1
        for value in unique_values:
            duplicate_indices = np.where(data == value)[0]
            for i in range(len(duplicate_indices)):
                data[duplicate_indices[i]] += (i % 2) * add_value

    mean = np.mean(data)
    std = np.std(data)
    z_scores_data = (data - mean) / std

    return z_scores_data

def z_score_standard(data):
    # Create a copy of the data to avoid modifying the original array
    data = np.copy(data)

    # Create conditional for same value
    unique_values, value_counts = np.unique(data, return_counts=True)
    if np.all(value_counts > 1):
        # Add 0.1 to every other occurrence of the duplicate values
        add_value = 0.1
        for value in unique_values:
            duplicate_indices = np.where(data == value)[0]
            for i in range(len(duplicate_indices)):
                data[duplicate_indices[i]] += (i % 2) * add_value

    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std

    # Scale z-scores to 0-1
    min_z = np.min(z_scores)
    max_z = np.max(z_scores)
    scaled_z = (z_scores - min_z) / (max_z - min_z)
    return scaled_z

def z_score_DeStandardization(data, original_data):
    # Create a copy of the data to avoid modifying the original array
    data = np.copy(data)

    # Create conditional for same value
    unique_values, value_counts = np.unique(original_data, return_counts=True)
    if np.all(value_counts > 1):
        # Add 0.1 to every other occurrence of the duplicate values
        add_value = 0.1
        for value in unique_values:
            duplicate_indices = np.where(original_data == value)[0]
            for i in range(len(duplicate_indices)):
                original_data[duplicate_indices[i]] += (i % 2) * add_value

    mean = np.mean(original_data)
    std = np.std(original_data)
    z_scores = (data * std) + mean
    return z_scores

def calculate_adjusted_r_squared(y_actual, y_predicted, n, k):
    
    # Hitung nilai R-squared
    r_squared = r2_score(y_actual, y_predicted)
    
    # Hitung Adjusted R-squared
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
    
    return adjusted_r_squared

def Test_kalkulasi():
    predict = np.array([32.9, 32.7, 32.6, 33.1, 32.9, 32.7, 32.6, 33.1, 32.9, 32.7, 32.6, 33.1, 32.9, 32.7, 32.6, 33.1, 32.9, 32.7, 32.6, 33.1, 32.9, 32.7, 32.6, 33.1, 32.9, 32.7, 32.6, 33.1, 32.9, 32.7])
    actual = np.array([33, 32.8, 31.4, 33.2, 33.4, 32.8, 33.6, 33.4, 32.6, 33, 33.5, 32.7, 33.2, 32, 32.2, 33.2, 33.4, 32, 32.4, 34, 33.8, 32.6, 33, 32.5, 32.2, 32.1, 32.1, 32.1, 32.2, 32])

    print(predict, "prediksi")

    n = len(actual)
    r = (n*(sum(actual*predict)) - sum(actual)*sum(predict)) / math.sqrt((n*sum(actual**2) - sum(actual)**2)*(n*sum(predict**2) - sum(predict)**2))

    r2 = r**2
    #r2 round 2 decimal
    r2 = round(r2,2)
    adjust_r2 = 1 - (1-r2)*(n-1)/(n-1-1)

    print("r2 nya bwang : ", r2)
    print("adjusted r2 nya bwang : ", adjust_r2)





    # pred = z_score(x)
    # test = z_score(test)
    # pred = x
    # test = test
    # pred = np.array(pred).flatten().__abs__().round(1)
    # test = np.array(test).flatten()

    # print("Hasil Z-score")
    # print(pred, "prediksi")
    # print(test, "test")
    # n = len(test)
    # k = 1

    # adjusted_r2_score = calculate_adjusted_r_squared(test, x, n, k)

    # # mae = np.mean(np.abs(pred - test))
    # # mape = np.mean((np.abs(pred - test)/np.abs(test))*100)
    # # mse = np.square(np.subtract(test,pred)).mean()
    # r2 = r2_score(test, x)
    # # rmse = math.sqrt(mse)

    # # print("Ini hasilnya")
    # # print(mae, "mae")
    # # print(mape, "mape")
    # # print(mse, "mse")
    # print(r2, "r2 nya bwang")
    # print(adjusted_r2_score, "adjusted r2 nya bwang")
    # print(rmse, "rmse")


def Prediction(df, seasonal, trend, periods, slevel, stren, sseasonal, start, end):
    low = df.index.get_loc(start)
    high = df.index.get_loc(end)

    train, test = df[1:high], df[low:high]

    # data = train.values
    # x = z_score(data)

    model = ExponentialSmoothing(train,
                                seasonal=seasonal,
                                trend=trend,
                                seasonal_periods=periods)
    model_fit = model.fit(smoothing_level=slevel, 
                        smoothing_trend=stren, 
                        smoothing_seasonal=sseasonal)
    
    # print(low, high, "ini low high")
    predictions = model_fit.predict(start=low, end= high-1)

    # predictions = z_score_DeStandardization(predictions, data)

    # print(predictions, "Hasil Prediksi")
    predictions = np.array(predictions).flatten().__abs__().round(1)
    test = np.array(test).flatten().__abs__().round(1)

    # ## Implement Z-score for error calculation
    z_predictions = z_score_calculation(predictions)
    z_test = z_score_calculation(test)


    mae = np.mean(np.abs(z_predictions - z_test))
    mape = np.mean((np.abs(z_predictions - z_test)/np.abs(test))* 100)
    mse = np.square(np.subtract(z_test,z_predictions)).mean()
    r2 = r2_score(test, predictions)
    rmse = math.sqrt(mse)

    return {
        'predictions' : predictions.tolist(),
        'mape' : mape,
        'mae' : mae,
        'mse' : mse,
        'r2' : r2,
        'rmse' : rmse
    }

def Forecast(df, seasonal, trend, periods, slevel, stren, sseasonal, end, fore, do_zscore):
    high = df.index.get_loc(end)
    train = df[0:high]
    # print(train, "ini train")
    # print(high, "ini high")
    # print(end, "ini end")
    #print head
    # print(train.tail())
    # print(high, "ini high")

    # # implement z-score for train data
    # data = train.values
    # x = z_score(data)
    # data = []
    if do_zscore == True:
        data = train.values
        x = z_score_standard(data)
    else : 
        x = train
    model = ExponentialSmoothing(x,
                                seasonal=seasonal,
                                trend=trend,
                                seasonal_periods=periods)
    model_fit = model.fit(smoothing_level=slevel, 
                        smoothing_trend=stren, 
                        smoothing_seasonal=sseasonal)
    # Forecast data z-score
    forecast = model_fit.forecast(steps=fore)
    if do_zscore == True:
        forecast = z_score_DeStandardization(forecast,train.values)
        forecast = np.array(forecast).flatten().__abs__().round(1)
    else : 
        forecast = np.array(forecast).flatten().__abs__().round(1)    
    #  z-score destandardization for forecast data  
    # forecast = z_score_DeStandardization(forecast, data)

    # forecast = np.array(forecast).flatten().__abs__().round(1)
    # print(forecast, "ini hasil forecastnya bwang : ")
    return forecast.tolist()

def fwiCalculation(Temp, rh, wind, rainfall):
    
    fwi_VALUE = []
    bui_VALUE = []
    isi_VALUE = []
    dmc_VALUE = []
    dc_VALUE = []
    ffmc_VALUE = []


    ## Set initial Value (Van wagner )
    # dmc_prev = 6
    # dc_prev = 15
    # FFMC_prev = 85 
    ## Set initial value (Turner and Lawson)
    dmc_prev = 2 * rainfall[0]
    dc_prev = 5 * rainfall[0]
    FFMC_prev = 85

    

    for i in range(len(Temp)):
        #set current value
        current_temp = Temp[i]
        current_rh = rh[i]
        current_wind = wind[i]
        current_rainfall = rainfall[i]
        #convert wind from m/s to km/h
        if current_wind > 10 : 
            current_wind = 10
        windkmh = current_wind * 3.6

        ## FFMC SECTION
        m_prev = 147.2 * (101.0 - FFMC_prev) / (59.5 + FFMC_prev)
        # Calculate mo
        if current_rainfall > 0.5:
            rf = current_rainfall - 0.5
            if m_prev > 150.0:
                mo = m_prev + (42.5 * rf * math.exp(-100.0 / (251.0 - m_prev)) * (1.0 - math.exp(-6.93 / rf))) + (0.0015 * (m_prev - 150.0) ** 2) * math.sqrt(rf)
                if mo > 250.0:
                    mo = 250.0
            else:
                mo = m_prev + 42.5 * rf * math.exp(-100.0 / (251.0 - m_prev)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mo = m_prev

        # Calculate Ed
        Ed = 0.942 * (current_rh ** 0.679) + (11.0 * math.exp((current_rh - 100.0) / 10.0)) + 0.18 * (21.1 - current_temp) * (1.0 - 1.0 / math.exp(0.115 * current_rh))

        # Calculate m
        if Ed > mo:
            Ew = 0.618 * (current_rh ** 0.753) + (10.0 * math.exp((current_rh - 100.0) / 10.0)) + 0.18 * (21.1 - current_temp) * (1.0 - 1.0 / math.exp(0.115 * current_rh))
            if mo < Ew:
                k1 = 0.424 * (1.0 - ((100.0 - current_rh) / 100.0) ** 1.7) + (0.0694 * math.sqrt(windkmh)) * (1.0 - ((100.0 - current_rh) / 100.0) ** 8)
                kw = k1 * (0.581 * math.exp(0.0365 * current_temp))
                m = Ew - (Ew - mo) * 10 **(-kw)
            else:
                m = mo
        else:
            k0 = 0.424 * (1.0 - ((current_rh / 100.0) ** 1.7)) + (0.0694 * math.sqrt(windkmh)) * (1.0 - ((current_rh / 100.0)) ** 8)
            kd = k0 * (0.581 * math.exp(0.0365 * current_temp))
            m = Ed + (mo - Ed) * 10**(-kd)

        # Calculate FFMC
        FFMC = (59.5 * (250.0 - m)) / (147.2 + m)
        ## DMC SECTION
            
        #Set default Day Lenght = 12 for indonesia
        Le = 12
        # Calculate mth = K for DMC
        if current_temp < -1.1:
            current_temp = -1.1
            mth = 1.894 * (current_temp + 1.1) * (100 - current_rh) * Le * 10 ** (-6)
        else:
            mth = 1.894 * (current_temp + 1.1) * (100 - current_rh) * Le * 10 ** (-6)

        # Calculate dmc with rainfall effect
        if current_rainfall > 1.5:
            rw = 0.92 * current_rainfall - 1.27
            wmi = 20.0 + math.exp(5.6348 - ((dmc_prev/ 43.43)))

            if dmc_prev <= 33.0:
                b = 100.0 / (0.5 + 0.3 * dmc_prev)
            elif dmc_prev <= 65.0:
                b = 14.0 - 1.3 * math.log(dmc_prev)
            else:
                b = 6.2 * math.log(dmc_prev) - 17.2

            #MRT Section
            wmr = wmi + 1000 * rw / (48.77 + b * rw)
            pr = 244.72-43.43 * math.log(wmr - 20.0)
            if pr < 0.0:
                pr = 0.0
            dmc = pr + 100.0 * mth
          
        else:
            pr = dmc_prev
            dmc = pr + 100.0 * mth
        
        ## DC SECTION
        ## Calculate DC
        if current_rainfall > 2.8:
            Pd = 0.83 * current_rainfall - 1.27
            ## Calculate Qprev
            Qprev = 800 * math.exp(-dc_prev / 400)

            ## Calculate Qrt
            Qrt = Qprev + 3.937 * Pd

            ## Calculate DCrt 
            Dcrt = 400 * math.log(800 / Qrt)

            if Dcrt <= 0:
                Dcrt = 0
        else:
            Dcrt = dc_prev

        ## Calculate V
        Lf = 12 #set day length default value for indonesia
        if current_temp < -2.8:
            current_temp = -2.8
        V = 0.36 * (current_temp + 2.8) + Lf
        if V <= 0:
            V = 0.0

        ## Calculate DC
        DC = Dcrt + 0.5 * V
        
        ## BUI SECTION
        buiprev = 0.4 * DC 
        if dmc <= buiprev :
            bui = 0.8 * (dmc * DC) / (dmc + 0.4 * DC)
        else:
            bui = dmc - (1 - (0.8 * DC / (dmc + 0.4 * DC))) * (0.92 + (0.0114 * dmc) ** 1.7)
        
        ## ISI Section
        mFuel = 147.2 * (101.0 - FFMC) / (59.5 + FFMC)
        Fu = math.exp(0.05039 * windkmh)
        Ff = 91.9 * math.exp(-0.1386 * mFuel) * (1.0 + (mFuel ** 5.31) / (4.93 * (10 ** 7)))
        isi = Ff * Fu* 0.208

        ## FWI  SECTION
        if bui <= 80.0:
            fD = 0.626 * (bui ** 0.809) + 2.0
            bScale = 0.1 * isi * fD
            if bScale <= 1.0:
                fwi = bScale
            else:
                fwi = math.exp(2.72 * (0.434 * math.log(bScale)) ** 0.647)
        else:
            fD = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
            bScale = 0.1 * isi * fD
            if bScale <= 1.0:
                fwi = bScale
            else:
                fwi = math.exp(2.72 * (0.434 * math.log(bScale)) ** 0.647)

        FFMC_prev = FFMC
        dmc_prev = dmc
        dc_prev = DC

        ## Append value round to 1 decimal Section
        ffmc_VALUE.append(round(FFMC,1))
        dmc_VALUE.append(round(dmc,1))
        dc_VALUE.append(round(DC,1))
        bui_VALUE.append(round(bui,1))
        isi_VALUE.append(round(isi,1))
        fwi_VALUE.append(round(fwi,1))

        #return ffmc, dmc, dc, bui, isi, fwi round to 1 decimal
    return ffmc_VALUE, dmc_VALUE, dc_VALUE, bui_VALUE, isi_VALUE, fwi_VALUE

def calculate_fwi_list(temperature_list, humidity_list, wind_list, rainfall_list):
    # ## NOTEEE : Data dari canada, rumus windkmh dari line 175 ( windkmh = current_wind)
    # temperature_list = [33.0, 32.8, 32.6, 32.5, 32.4, 32.0, 33.0, 32.4, 32.0, 32.0, 32.0, 32.0, 31.2, 31.0, 31.4, 31.2, 31.4, 31.8, 31.4, 31.4, 31.0, 30.5, 31.8, 31.0, 28.0, 30.5, 30.5, 31.2, 30.2, 28.5]
    # humidity_list = [85, 82, 81, 81, 82, 80, 80, 79, 80, 80, 81, 80, 81, 81, 78, 79, 80, 79, 80, 77, 78, 76, 76, 77, 80, 80, 76, 78, 80, 78]
    # rainfall_list = [5.0, 5.0, 2.0, 2.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 3.8, 6.0, 6.0, 10.5, 6.0, 6.0, 7.0, 7.0, 6.0,6.0, 10.0, 10.0, 12.6, 10.7, 10.1, 18.5]
    # wind_list = [6, 5, 4, 5, 5, 5, 5, 4, 4, 5, 5, 6, 6, 5, 4, 5, 6, 5, 5, 4, 4, 5, 4, 5, 5, 5, 3, 4, 6, 4]
    data_list = []

    ffmc_list, dmc_list, dc_list, bui_list, isi_list, fwi_list = fwiCalculation(temperature_list, humidity_list, wind_list, rainfall_list)
    for i in range(len(ffmc_list)):
        data_list.append({
            'ffmc': ffmc_list[i],
            'dmc': dmc_list[i],
            'dc': dc_list[i],
            'bui': bui_list[i],
            'isi': isi_list[i],
            'fwi': fwi_list[i]
        })

    return data_list

def dataProcessing(data, start, end,freq='D'):
    # def pre_Fix_data(data) :
    #     data = data.replace('', np.nan)
    #     data = data.replace(['0'], np.nan)
    #     data = data.replace(['8888', ''], np.nan)
    #     data = data.astype(float)
    #     data = data.fillna(method='ffill').fillna(method='bfill')         
    #     return data

    def pre_Fix_data(method,data) :
        # data = data.replace('', np.nan)
        # # data = data.replace(['0'], np.nan)
        # data = data.replace(['8888', ''], np.nan)
        # data = data.astype(float)
        # data = data.fillna(data.mean())
        if method == True :
            data = data.replace('', np.nan)
            # data = data.replace(['0'], np.nan)
            data = data.replace(['8888', ''], np.nan)
            data = data.astype(float)
            data = data.fillna(method='ffill').fillna(method='bfill')
        else :
            data = data.replace('', np.nan)
            data = data.replace(['0'], np.nan)
            data = data.replace(['8888', ''], np.nan)
            data = data.astype(float)
            data = data.fillna(data.mean())
        # data = data.fillna(data.mean())
        return data
    
    index = pd.date_range(start, end, freq=freq)
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df.rename(columns={'Tanggal':'Date','Tx':'Temperature','RH_avg':'Humidity','ff_x':'Wind','RR':'Rainfall'})
    
    #filter data by range date start and end
    df = df.drop(df.index[0]) 
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # df_filtered = df[df['Date'].between(start, end)]
 
    parameters = []
    parameters_names = ['Temperature', 'Humidity', 'Wind', 'Rainfall']

    for param_name in parameters_names : 
        param_data = df[['Date', param_name]].set_index('Date')

        if param_name == 'Rainfall' :
            method = True
            param_data = pre_Fix_data(method,param_data)
        else :
            method = False
            param_data = pre_Fix_data(method,param_data)
        # param_data = pre_Fix_data(param_data).resample('D').mean()
        param_list = param_data[param_name].tolist()

        #conditional argument for parameter
        if param_name == 'Temperature' : 
            x = None
            y = None
            period = 4
            alpha = 0.9
            beta = 0.1
            gamma = 0.1
        elif param_name == 'Humidity' :
            x = None
            y = None
            period = 4
            alpha = 0.9
            beta = 0.4
            gamma = 0.1
        elif param_name == 'Wind' :
            x = 'multiplicative'
            y = None
            period = 4
            alpha = 0.9
            beta = 0.1
            gamma = 0.1
        else :
            x = 'additive'
            y = None
            period = 4
            alpha = 0.9
            beta = 0.1
            gamma = 0.1
        parameters.append({
            'name': param_name,
            'data': param_data,
            'trend' : x,
            'seasonal' : y,
            'periode' : period,
            'alpha' : alpha,
            'beta' : beta,
            'gamma' : gamma
            # 'trend': None if param_name in ['Temperature','Humidity'] else 'multiplicative' if param_name == 'Wind' else 'additive' if param_name == 'Rainfall' else None,
            # 'seasonal': None,
        })

        globals()[f'{param_name.lower()}_list'] = param_list

    predict_result = []
    error_result = []
    # date list from start to end
    date_list = pd.date_range(start, end, freq=freq).tolist()
    # Looping for prediction
    for param in parameters:
        result = Prediction(param['data'], param['seasonal'], param['trend'],param['periode'],param['alpha'] ,param['beta'], param['gamma'],start, end)
        predict_result.append({param['name']: result['predictions']})
        error_result.append({
            param['name']: {
                'MAPE': result['mape'],
                'MAE': result['mae'],
                'MSE': result['mse'],
                'RMSE': result['rmse'],
                'R2': result['r2']
            }
    })

    ## Fuzzy Universe
    def fuzzy(value):
        result=[]



        if value <= 1 :
            result = [1, 0, 0, 0]
        elif value > 1 and value <= 6 :
            result = [0, 1, 0, 0]
        elif value > 6 and value <= 13 :
            result = [0, 0, 1, 0]
        else :
            result = [0, 0, 0, 1]

        # plt.plot(fwi.universe, fwi['biru'].mf, 'b', linewidth=1.5, label='Biru')
        # plt.plot(fwi.universe, fwi['hijau'].mf, 'g', linewidth=1.5, label='Hijau')
        # plt.plot(fwi.universe, fwi['kuning'].mf, 'y', linewidth=1.5, label='Kuning')
        # plt.plot(fwi.universe, fwi['merah'].mf, 'r', linewidth=1.5, label='Merah')
        # # plt.show()
        return result
    
    # Get Prediction result from each parameter list 
    temperature_prediction = predict_result[0]['Temperature']
    humidity_prediction = predict_result[1]['Humidity']
    wind_prediction = predict_result[2]['Wind']
    rainfall_prediction = predict_result[3]['Rainfall']


    #implement to calculate fwi from all parameters
    fwi_values = calculate_fwi_list(temperature_prediction, humidity_prediction, wind_prediction, rainfall_prediction)

    # Get data from fwi_values
    data_ffmc = [item['ffmc'] for item in fwi_values]
    data_dmc = [item['dmc'] for item in fwi_values]
    data_dc = [item['dc'] for item in fwi_values]
    data_bui = [item['bui'] for item in fwi_values]
    data_isi = [item['isi'] for item in fwi_values]
    data_fwi = [item['fwi'] for item in fwi_values]

    # Fuzzy for prediction
    fuzzy_result = []
    for i in range(len(fwi_values)): 
        result = fuzzy(data_fwi[i])
        fuzzy_result.append({
            'FFMC Value' : data_ffmc[i],
            'DMC Value' : data_dmc[i],
            'DC Value' : data_dc[i],
            'BUI Value' : data_bui[i],
            'ISI Value' : data_isi[i],
            'FWI Value' : data_fwi[i], 
            'Fuzzy' : result})

    # List for result
    data_result = []

    # Looping for Prediction
    for i in range(len(predict_result[0]['Temperature'])):
        result = {
            'Date': date_list[i],
            'Prediction': {
                'Temp': predict_result[0]['Temperature'][i],
                'Humidity': predict_result[1]['Humidity'][i],
                'Wind': predict_result[2]['Wind'][i],
                'Rainfall': predict_result[3]['Rainfall'][i]
            },
            'Result': {
                'Category': {
                    'ffmc': fwi_values[i]['ffmc'],
                    'dmc': fwi_values[i]['dmc'],
                    'dc': fwi_values[i]['dc'],
                    'bui': fwi_values[i]['bui'],
                    'isi': fwi_values[i]['isi'],
                    'fwi': fwi_values[i]['fwi']
                },
                'Fuzzy': fuzzy_result[i]['Fuzzy']
            }
        }
        data_result.append(result)

    response = {
        'Prediksi_Error' : error_result,
        'Data_Result' : data_result,
    }
    return response

## Data processing for forecast

def ForecastProcessing(data, fore,freq='D'):
    def pre_Fix_data(method,data) :
        # data = data.replace('', np.nan)
        # # data = data.replace(['0'], np.nan)
        # data = data.replace(['8888', ''], np.nan)
        # data = data.astype(float)
        # data = data.fillna(data.mean())
        if method == True :
            data = data.replace('', np.nan)
            # data = data.replace(['0'], np.nan)
            data = data.replace(['8888', ''], np.nan)
            data = data.astype(float)
            data = data.fillna(method='ffill').fillna(method='bfill')
        else :
            data = data.replace('', np.nan)
            data = data.replace(['0'], np.nan)
            data = data.replace(['8888', ''], np.nan)
            data = data.astype(float)
            data = data.fillna(data.mean())
        # data = data.fillna(data.mean())
        return data
    
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    # df = df[:-periods]
    df = df.rename(columns={'Tanggal':'Date','Tx':'Temperature','RH_avg':'Humidity','ff_x':'Wind','RR':'Rainfall'})
    
    #filter data by range date start and end
    df = df.drop(df.index[0]) 
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # df_filtered = df[df['Date'].between(start, end)]

 
    parameters = []
    parameters_names = ['Temperature', 'Humidity', 'Wind', 'Rainfall']

    for param_name in parameters_names : 
        param_data = df[['Date', param_name]].set_index('Date')
        if param_name == 'Rainfall' :
            method = True
            param_data = pre_Fix_data(method,param_data)
        else :
            method = False
            param_data = pre_Fix_data(method,param_data)
        #print 7 data last from param_data
        
        # print("Data parameter")
        # print(param_data.tail(30))
        # Test_kalkulasi()
        param_list = param_data[param_name].tolist()

        # #conditional argument Property for parameter
        if param_name == 'Temperature' :
            x = 'multiplicative'
            y = 'additive'
            period = 4
            alpha = 0.6
            beta = 0.30000000000000004
            gamma = 0.1

            #256
            # grid_search(param_data)
        elif param_name == 'Humidity' :
            x = 'additive'
            y = 'multiplicative'
            period = 12
            alpha = 0.2
            beta = 0.4
            gamma = 0.1   
            # grid_search(param_data)
        elif param_name == 'Wind' :
            x = None
            y = 'multiplicative'
            period = 12
            alpha = 0.9
            beta = 0.1
            gamma = 0.2
            # grid_search(param_data)
        else :
            x = 'additive'
            y = None
            period = 4
            alpha = 0.5
            beta = 0.1
            gamma = 0.1
            # grid_search(param_data)
            #'additive'zz
            #'multiplicative'

        parameters.append({
            'name': param_name,
            'data': param_data,
            'trend' : x,
            'seasonal' : y,
            'periode' : period,
            'alpha' : alpha,
            'beta' : beta,
            'gamma' : gamma
        })

        globals()[f'{param_name.lower()}_list'] = param_list

    

    #get last date from dataset
    last_date = df['Date'].max()
    start_fore = last_date + pd.DateOffset(days=1)
    end_fore = last_date + pd.DateOffset(days=fore)
    date_fore = pd.date_range(start_fore, end_fore, freq='D').tolist()

    # List for result
    forecast_result = []
    # Looping for prediction
    for param in parameters:
        if param['name'] == 'rainfall' :
            do_z_score = True
            forecast = Forecast(param['data'], param['seasonal'], param['trend'],param['periode'],param['alpha'] ,param['beta'], param['gamma'], last_date,fore, do_z_score)
        else : 
            do_z_score = False
            forecast = Forecast(param['data'], param['seasonal'], param['trend'],param['periode'],param['alpha'] ,param['beta'], param['gamma'], last_date,fore, do_z_score)
        # forecast = Forecast(param['data'], 'additive', 'additive', 4, 0.3, 0.3, 0.3, last_date, fore)
        # Test_kalkulasi()
        forecast_result.append({param['name']: forecast})

    ## Fuzzy Universe
    def fuzzy(value):
        result=[]



        if value <= 1 :
            result = [1, 0, 0, 0]
        elif value > 1 and value <= 6 :
            result = [0, 1, 0, 0]
        elif value > 6 and value <= 13 :
            result = [0, 0, 1, 0]
        else :
            result = [0, 0, 0, 1]

        # plt.plot(fwi.universe, fwi['biru'].mf, 'b', linewidth=1.5, label='Biru')
        # plt.plot(fwi.universe, fwi['hijau'].mf, 'g', linewidth=1.5, label='Hijau')
        # plt.plot(fwi.universe, fwi['kuning'].mf, 'y', linewidth=1.5, label='Kuning')
        # plt.plot(fwi.universe, fwi['merah'].mf, 'r', linewidth=1.5, label='Merah')
        # # plt.show()
        return result
    


    ## Get Forecast result from each parameter list
    temperature_forecast = forecast_result[0]['Temperature']
    humidity_forecast = forecast_result[1]['Humidity']
    wind_forecast = forecast_result[2]['Wind']
    rainfall_forecast = forecast_result[3]['Rainfall']

    #implement to calculate fwi from all parameters
    fwi_forecast = calculate_fwi_list(temperature_forecast, humidity_forecast, wind_forecast, rainfall_forecast)

    # Get data from fwi_forecast
    data_ffmc_fore = [item['ffmc'] for item in fwi_forecast]
    data_dmc_fore = [item['dmc'] for item in fwi_forecast]
    data_dc_fore = [item['dc'] for item in fwi_forecast]
    data_bui_fore = [item['bui'] for item in fwi_forecast]
    data_isi_fore = [item['isi'] for item in fwi_forecast]
    data_fwi_fore = [item['fwi'] for item in fwi_forecast]
    
    # Fuzzy for forecast
    fuzzy_forecast = []
    for i in range(len(fwi_forecast)):
        result = fuzzy(data_fwi_fore[i])
        fuzzy_forecast.append({
            'FFMC Value' : data_ffmc_fore[i],
            'DMC Value' : data_dmc_fore[i],
            'DC Value' : data_dc_fore[i],
            'BUI Value' : data_bui_fore[i],
            'ISI Value' : data_isi_fore[i],
            'FWI Value' : data_fwi_fore[i], 
            'Fuzzy' : result})

    # List for result
    forecast_data = []


    # Looping for Forecast
    for i in range(len(forecast_result[0]['Temperature'])):
        result = {
            'Date': date_fore[i],
            'Prediction': {
                'Temp': forecast_result[0]['Temperature'][i],
                'Humidity': forecast_result[1]['Humidity'][i],
                'Wind': forecast_result[2]['Wind'][i],
                'Rainfall': forecast_result[3]['Rainfall'][i]
            },
            'Result': {
                'Category': {
                    'ffmc': fwi_forecast[i]['ffmc'],
                    'dmc': fwi_forecast[i]['dmc'],
                    'dc': fwi_forecast[i]['dc'],
                    'bui': fwi_forecast[i]['bui'],
                    'isi': fwi_forecast[i]['isi'],
                    'fwi': fwi_forecast[i]['fwi']
                },
                'Fuzzy': fuzzy_forecast[i]['Fuzzy']
            }
        }
        forecast_data.append(result)
   


    response = {
        'Data_Result' : forecast_data,
    }
    return response


## API for Prediction
@app.route('/api')
def get_credentials():
    sheetName = request.args.get('sheetName')
    worksheetName = request.args.get('worksheetName')
    start = request.args.get('start')
    end = request.args.get('end')
    
    scope_app =['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive'] 
    cred = ServiceAccountCredentials.from_json_keyfile_name('token.json',scope_app)  # type: ignore
    client = gspread.authorize(cred)
    sheet = client.open(sheetName)
    worksheet = sheet.worksheet(worksheetName)
    data = worksheet.get_all_values()

    return dataProcessing(data, start, end)# type: ignore

## API for Forecast
@app.route('/api/forecast')
def get_forecast():
    sheetName = request.args.get('sheetName')
    worksheetName = request.args.get('worksheetName')
    fore = request.args.get('fore')

    scope_app =['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    cred = ServiceAccountCredentials.from_json_keyfile_name('token.json',scope_app)  # type: ignore
    client = gspread.authorize(cred)
    sheet = client.open(sheetName)
    worksheet = sheet.worksheet(worksheetName)
    data = worksheet.get_all_values()

    return ForecastProcessing(data,int(fore))# type: ignore

if __name__ == '__main__':
    app.run(debug=True, port=3000)
