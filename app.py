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

def Prediction(df, seasonal, trend, periods, slevel, stren, sseasonal):
    print(len(df))
    low = 1451
    high = 1471

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
    
    ## Calculate FFMC
    prev_ffmc = 85
    prev_dmc = 6
    prev_dc = 15

    ### FFMC AREA

    ## Calculate FFMC 
    mrprev = 147.2 * ((101 - prev_ffmc) / (59.5 + prev_ffmc))

    ## Calculate pF
    if rainfall <= 0.5 :
        pF =  rainfall - 0.5
    else :
        pF = rainfall
    ## Calculate Mrt
    if mrprev <= 150 :
        Mrt = mrprev + 42.5 * pF * math.exp((251-mrprev)/100) * (1 - math.exp(pF/6.93))
    elif mrprev > 150 :
        term1 = 42.5 * pF * math.exp((251 - mrprev)/100) * (1 - math.exp(pF/6.93))
        term2 = 0.0015 * ((mrprev - 150) ** 2) * (math.sqrt(pF))
        if (251 - mrprev) / 100 < 0:
            Mrt = mrprev
        else:
            Mrt = mrprev + term1 + term2

    ## Calculate Ed
    Ed = 0.942 * (humidity ** 0.679) + (11 * math.exp((humidity - 100) / 10)) + 0.18 * (21.1 - temperature) * (1 - math.exp(1/(0.115 * humidity)))

    ## Calculate Kd
    
    if Ed <= mrprev : 
        Ko = 0.424 * (1 - (humidity/ 100) ** 1.7) + (0.0694 * math.sqrt(wind)) * (1 - ((humidity) / 100) ** 8)
        Kd = Ko * (0.581 * math.exp(0.0365 * temperature))
        m = Ed + (mrprev - Ed ) * 10 ** (1/Kd)
        print(m,"m KD")
    ## Calculate Ew
    else :
        Ew = 0.618 * (humidity ** 0.753) + (10 * math.exp((humidity - 100) / 10)) + 0.18 * (21.1 - temperature) * (1 - math.exp(-0.115 * humidity))

    ## Calculate conditional K1
        if Ew > mrprev :
            K1 = 0.424 * (1 - ((100-humidity / 100) ** 1.7)) + (0.0694 * math.sqrt(wind)) * (1 - ((100-humidity / 100) ** 8))
            Kw = K1 * (0.581 * math.exp(0.0365 * temperature))
            m = Ew - (Ew - mrprev) * 10 ** (1/Kw)
        else :
            m = mrprev
    ## Calculate FFMC
    FFMC = 59.5 * ((250 - m )/ (147.2 + m))



    ### DMC AREA
    ## Calculate Pe 
    Dmcrt = 0.0
    if rainfall > 1.5 :
        Pe = 0.92 * rainfall - 1.27

        ## Calculate MtPrev 
        MtPrev = 20 + (math.exp(5.6348 - (prev_dmc / 43.43 )))

        ## Calculate b 
        b = 0.0
        if prev_dmc <= 33 :
            b = 100 / (0.5 + 0.3 * prev_dmc)
        elif prev_dmc > 33 and prev_dmc <= 65 :
            b = 14 - 1.3 * math.log(prev_dmc)
        elif prev_dmc > 65 :
            b = 6.2 * math.log(prev_dmc) - 17.2
        
        ## Calculate Mrt
        Mrt = MtPrev + 1000 * Pe / (48.77 + b * Pe)

        ## Calculate Dmcrt
        Dmcrt = 244.72 - 43.43 * math.log(Mrt - 20)
        if(Dmcrt < 0):
            Dmcrt = 0

    ## Calculate K 
    Le = 12
    K = 1.894 * (temperature + 1.1) * (100 - humidity) * Le * 10 ** (-6)
    ### calculate DMC
    DMC = 0.0
    if rainfall <= 1.5 :
        DMC = prev_dmc + 100 * K 
    elif rainfall > 1.5 :
        DMC = Dmcrt + 100 * K 

    ### DC AREA
    Dcrt = 0.0
    if rainfall > 2.8 :
        Pd = 0.83 * rainfall - 1.27
    
        ## Calculate Qprev
        Qprev = 800 * math.exp(-prev_dc / 400)

        ## Calculate Qrt
        Qrt = Qprev + 3.937 * Pd

        ## Calculate DCrt
        Dcrt = 400 * math.log(800 / Qrt)

        if Dcrt < 0 :
            Dcrt = 0

    ## Calculate V
    Lf = 12
    if(temperature< -2.8):
        temperature = -2.8;
        V = 0.36 * (temperature + 2.8) + Lf 
    else:
        V = 0.36 * (temperature + 2.8) + Lf

    ## Calculate DC
    DC = 0
    if rainfall <= 2.8 :
        DC = prev_dc + 0.5 * V
    elif rainfall > 2.8 :
        DC = Dcrt + 0.5 * V


    ### ISI AREA
    decimal.getcontext().prec = 2
    ## Calculate m
    m = 147.2 * ((101 - FFMC) / (59.5 + FFMC))
    ## Calculate fU 
    fU = math.exp(0.05039 * wind)
    ## Calculate fF
    tes = 1 + ((m ** 5.31)/ (4.93 * (10**(7))))
    fF = 91.9*math.exp(1/(0.1386*m))*tes
    ## Calculate ISI
    ISI = 0.208 * fU * fF


    ### BUI AREA
    if DMC <= 0.4 * DC :
        BUI = 0.8 * (DMC * DC) / (DMC + 0.4 * DC)
    else :
        BUI = DMC - (1 - (0.8 * DC / (DMC + 0.4 * DC))) * (0.92 + (0.0114 * DMC) ** 1.7)    
    ### FWI AREA
    ## Calculate fD 
    if BUI <= 80 :
        fD = 0.626 * (BUI ** 0.809) + 2
    else :
        fD = 1000 / (25 + 108.64 * math.exp(-0.023 * BUI))
    ## Calculate BScale
    BScale = abs(0.1 * ISI * fD)
    ## Calculate FWI
    FWI = 0.0
    if BScale > 1 :
        FWI = math.exp(2.72 * (0.434 * (math.log(BScale)) ** 0.647))
    elif BScale <= 1 :
        FWI = BScale
    #Return with round 2 decimalsss
    return round(FWI,2)

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
    df = df.rename(columns={'Tanggal':'Date','Tx':'Temperature','RH_avg':'Humidity','ff_avg':'Wind','RR':'Rainfall'})
    
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

    # Kabupaten Kotawaringin Barat versi baru
    temp = Prediction(temperature_data,None, None, 4, 0.9, 0.1, 0.1)
    humidity = Prediction(humidity_data,None, 'multiplicative', 4, 0.9, 0.1, 0.1)
    wind = Prediction(wind_data,None, 'additive', 4, 0.9, 0.1,0.1)
    rainfall = Prediction(rainfall_data,None, None, 4, 0.9, 0.1,0.1)


    predict_result = [
        {'Temperature' : temp},
        {'Humidity' : humidity},
        {'Wind' : wind},
        {'Rainfall' : rainfall}
    ]


    #Fuzzy Universe

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
    app.run(debug=True, port=3000)