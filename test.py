from flask import Flask, request, make_response, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import r2_score


import numpy as np

actual = np.array([33, 32.8, 31.4, 33.2, 33.4, 32.8, 33.6, 33.4, 32.6, 33, 33.5, 32.7, 33.2, 32, 32.2, 33.2, 33.4, 32.8, 33.4, 33.4, 32.6, 33.7, 33, 32.8, 33, 33.5, 32.5, 32.2, 32.2, 28.5])
predicted = np.array([32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32.7, 32])

# bulatkan nilai prediksi ke bawah atau ke atas
# predicted = np.floor(predicted)
# print(predicted)

n = len(actual)
r = (n*(sum(actual*predicted)) - sum(actual)*sum(predicted)) / math.sqrt((n*sum(actual**2) - sum(actual)**2)*(n*sum(predicted**2) - sum(predicted)**2))

print(r2_score)
r2 = r**2
print(r2)

adj_r2 = 1 - (1-r2)*(n-1)/(n-1-1)

print(adj_r2)


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
                k1 = 0.424 * (1.0 - (1.0 - current_rh / 100.0) ** 1.7) + (0.0694 * math.sqrt(windkmh)) * (1.0 - (1.0 - current_rh / 100.0) ** 8)
                kw = k1 * (0.581 * math.exp(0.0365 * current_temp))
                m = Ew - (Ew - mo) * math.exp(-kw)          
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
            if pr < 0.0:
                pr = 0.0
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



fwiCalculation([33,34], [80,81], [4,5], [1,2])