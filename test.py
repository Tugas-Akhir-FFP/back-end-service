from flask import Flask, request, make_response, jsonify
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


def arima_model(df, params): 
    df = df[params]
    df = np.array(df) 
    df = df.astype('float32')
    model = sm.tsa.arima.ARIMA(df, order=(25,1,10)) 
    result = model.fit(method_kwargs={'warn_convergence':False})
    forecast = result.forecast(steps=25) 
    Tanggal = pd.date_range('2019-12-31', periods=25, freq='D')
    Tanggal = Tanggal.tolist()
    Tanggal = np.array(Tanggal)
    forecast = forecast.tolist()
    forecast = np.array(forecast)
    forecast = forecast.astype('float32')
    response = {}
    if(params == 'Temperature'):
        response['Tanggal'] = Tanggal
    response[params] = forecast
    print(res)

def dataProcessing(data):
    index = pd.date_range('2015-01-01', '2019-12-31', freq='D')
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df[:-10]
    df  = df.rename(columns={'Tanggal':'Date','Tavg':'Temperature','RH_avg':'Humidity','ff_avg':'Wind','RR':'Rainfall'})
    #menentukan range data berdasarkan tanggal
    df = df.drop(df.index[0]) 
    df = df.drop(df.columns[0], axis=1)
    df = df.set_index(index) 
    df = df.replace('',math.nan) 
    df = df.replace('8888',math.nan)
    # df  = df.dropna()
    df = df.astype('float32')
    df = df.fillna(df.mean())
    df = df.loc['2019-01-01':'2019-12-31']
    # df = df['Temperature'].fillna(df['Temperature'].mean())
    
    # Scaling berbasis Median Absolute Deviation (MAD)
    print(df)
    # for i in range(3):
    #     arima_model(df,df.columns[i])
    # arima_model(df) 
def get_credentials():
    scope_app =['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive'] 
    cred = ServiceAccountCredentials.from_json_keyfile_name('token.json',scope_app) 
    client = gspread.authorize(cred)
    sheet = client.open('Riau-Kab.Kampar_2015-2019')
    worksheet = sheet.worksheet('Data Harian - Table')
    data = worksheet.get_all_values()
    dataProcessing(data)
if __name__ == '__main__':
  get_credentials()