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



app = Flask(__name__)
api = Api(app)
CORS(app)


def arima_model(df, param, periods): 
    df = df[param]
    df = np.array(df) 
    df = df.astype('float32')
    model = sm.tsa.arima.ARIMA(df, order=(periods,1,10)) 
    result = model.fit(method_kwargs={'warn_convergence':False})
    forecast = result.forecast(steps=25) 
    forecast = forecast.tolist()
    return forecast

def dataProcessing(data, periods): 
    index = pd.date_range('2015-01-01', '2019-12-31', freq='D')
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df[:-10]
    df  = df.rename(columns={'Tanggal':'Date','Tavg':'Temperature','RH_avg':'Humidity','ff_avg':'Wind','RR':'Rainfall'})
    df = df.fillna(0)
    df = df.drop(df.index[0]) 
    df = df.drop(df.columns[0], axis=1)
    df = df.set_index(index)
    df = df.replace('',math.nan)
    df = df.replace(8888,math.nan)
    df  = df.dropna()
    df = df.reset_index(drop=True)
    
    Tanggal = pd.date_range('2019-12-31', periods=periods, freq='D')
    Tanggal = Tanggal.tolist()
    response = {}
    response['Tanggal'] = Tanggal
    for i in range(3):
        response[df.columns[i]] = arima_model(df,df.columns[i], periods)
    return response
# Riau-Kab.Kampar_2015-2019
# Data Harian - Table
@app.route('/api')
def get_credentials():
    sheetName = request.args.get('sheetName')
    worksheetName = request.args.get('worksheetName')
    periods = request.args.get('periods')
    
    scope_app =['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive'] 
    cred = ServiceAccountCredentials.from_json_keyfile_name('token.json',scope_app) 
    client = gspread.authorize(cred)
    sheet = client.open(sheetName)
    worksheet = sheet.worksheet(worksheetName)
    data = worksheet.get_all_values()
    return dataProcessing(data, int(periods))
    

if __name__ == '__main__':
    app.run(debug=True,port=4000)
      