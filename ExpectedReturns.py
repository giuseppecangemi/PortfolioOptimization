#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: giuseppecangemi
"""

import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import seaborn as sns

#importo il dataframe da YahooFinance. Unicredit, BPM, IntesaSanPaolo.
df = data.DataReader(["UCG.MI", "BAMI.MI","ISP.MI"], "yahoo", start="2021/01/01", end="2022/12/03")
df.head()
#modifico il dataframe lasciando solo il prezzo di chiusura adj
df = df["Adj Close"]
#modifico nomi colonne:
for i in df.columns:
    if i == "UCG.MI":
        df.rename(columns = {'UCG.MI':'unicredit'}, inplace = True)
    elif i == "BAMI.MI": 
        df.rename(columns = {'BAMI.MI':'bpm'}, inplace = True)
    elif i == "ISP.MI": 
        df.rename(columns = {'ISP.MI':'intesa'}, inplace = True) 
  
#calcolo le differenze prime in % (returns giornalieri)        
df["uni_returns"] = ((df["unicredit"] - df["unicredit"].shift(1))/df["unicredit"].shift(1))*100
df["bpm_returns"] = ((df["bpm"] - df["bpm"].shift(1))/df["bpm"].shift(1))*100
df["int_returns"] = ((df["intesa"] - df["intesa"].shift(1))/df["intesa"].shift(1))*100

#creo histplot per osservare la distribuzine dei ritorni giornalieri
sns.histplot(df["uni_returns"], kde=True)
plt.title("UniCredit - Histogram of Daily Returns")

sns.histplot(df["bpm_returns"], color="darkorange", kde=True)
plt.title("Banco Popolare di Milano - Histogram of Daily Returns")

sns.histplot(df["int_returns"], kde=True, color="red")
plt.title("Intesa SanPaolo - Histogram of Daily Returns")

#calcolo varianza per le tre banche:
var_uni = df["uni_returns"].var()    
var_uni

var_bpm = df["bpm_returns"].var()
var_bpm

var_int = df["int_returns"].var()
var_int

#calcolo std.dev per le tre banche (normalizzo per l'anno moltiplicando x250):
stdv_uni = np.sqrt(var_uni * 250)
stdv_uni

stdv_bpm = np.sqrt(var_bpm * 250)
stdv_bpm

stdv_int = np.sqrt(var_int * 250)
stdv_int

#comparazione della deviazione standard per le tre banche:
plt.bar("UniCredit",stdv_uni)
plt.bar("BPM",stdv_bpm)
plt.bar("Intesa",stdv_int)
plt.title("Standard Deviation Comparation")

#covariance:

df["int_returns"].cov(df["uni_returns"])    
df["int_returns"].cov(df["bpm_returns"])

df["uni_returns"].cov(df["int_returns"])    
df["uni_returns"].cov(df["bpm_returns"])

df["bpm_returns"].cov(df["int_returns"])    
df["bpm_returns"].cov(df["uni_returns"])

#covariance matrix:
#creo nuovo dataframe per sviluppare covariance matrix
df_cov = pd.DataFrame()
df_cov = df[["unicredit","bpm","intesa"]].copy()
#cov matrix in log del tasso di ritorno giornaliero
cov_matrix = df_cov.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix

#plotto cov_matrix
sns.heatmap(cov_matrix, cmap="viridis",annot=True)
plt.title("Covariance Matrix")

#correlazione
corr_matrix = df_cov.pct_change().apply(lambda x: np.log(1+x)).corr()
corr_matrix

#plotto cov_matrix
sns.heatmap(corr_matrix, cmap="viridis",annot=True)
plt.title("Correlation Matrix")

#adesso assumo dei pesi (weight) random per i tre asset
#questi pesi sono la quota degli asset all'interno del portafoglio 
#la sommatoria di w che va da 1 a 3 è strettamente uguale a 1
w = [0.25, 0.25, 0.5]
#in questo modo sto dicendo che il mio portafoglio è formato dal 
#25% di unicredit, 25% di bpm e il 25% di intesa
#Per poter calcolare la varianza del portafoglio seguo una formula differente 
#sigma = sum_1_n sum_1_n sum_1_n w_uni w_bpm w_int cov(R_uni, R_bpm, R_int)
#Pertanto costruisco un dizionario dei pesi:

w = {"unicredit": 0.25, "bpm": 0.25, "intesa": 0.5}    

#dato che ho già calcolato la cov_matrix:

var_portafoglio = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()
var_portafoglio

#.mul è una funzione che ci permette di svolgere moltiplicazioni sugli oggetti di un dataframe
#con il codice (cov_matrix.mul(w, axis=0)) stiamo moltiplicando gli elementi 
#della matrice covarianza per i rispettivi pesi. 
#con .sum() sto sommando la riga di ogni asset (unicredit,bpm,intesa)
#ottenendo un solo risultato per asset che è la somma delle covarianze tra,
#ad esempio, unicredit e se stesso (varianza), unicredit-bpm, unicredit-intesa
#infine con l'ulteriore .sum() sommo le covarianza dei tre asset ottenendo un solo risultato
#che sarà la varianza del portafoglio (sigma = 0.0004348)

#ADESSO OTTENGO GLI EXPECTED RETURNS:
#pertanto mi basta moltiplicare la media dei rendimenti per il peso

daily_avg_uni = df["uni_returns"].mean()
daily_avg_bpm = df["bpm_returns"].mean()
daily_avg_int = df["int_returns"].mean()

for key, value in w.items():
    if key == "unicredit":
        exp_ret_uni = value*daily_avg_uni
        print("ExpectedReturn "+key + ": " +str(exp_ret_uni))
    elif key == "bpm":
        exp_ret_bpm = value*daily_avg_bpm
        print("ExpectedReturn "+key + ": " + str(exp_ret_bpm))
    elif key == "intesa":
        exp_ret_int = value*daily_avg_int  
        print("ExpectedReturn "+key + ": " + str(exp_ret_int))
        
expected_return_portafoglio = exp_ret_uni + exp_ret_bpm + exp_ret_int
expected_return_portafoglio

#I RISULTATI OTTENUTI SONO COMUNQUE RIFERITI AI RITORNI ATTESI GIORNALIERI
#SE VOGLIAMO SVILUPAPRE UNA DIFFERENTE ANALISI, PRENDENDO IN CONSIDERAZIONE UN ARCO TEMPORALE PIU GRANDE
#DOBBIAMO FARE IL RESAMPLE DEL NOSTRO DATAFRAME
#Ipotizziamo lo vogliamo fare mensile:
df_month = df[["uni_returns","bpm_returns","int_returns"]].copy()
df_month = df_month.resample('m').mean()
#come possiamo vedere il campione è passato da print(len(df_cov)) = 493 dati
# a print(len(df_month)) = 24 dati che sono esattamente i due anni di riferimento 
#decisi prima dello sviluppo dell'analisi

#df_month.mean().sum()

monthly_avg_uni = df_month["uni_returns"].mean()
monthly_avg_bpm = df_month["bpm_returns"].mean()
monthly_avg_int = df_month["int_returns"].mean()

for key, value in w.items():
    if key == "unicredit":
        exp_ret_uni = value*monthly_avg_uni
        print("ExpectedReturn "+key + ": " +str(exp_ret_uni))
    elif key == "bpm":
        exp_ret_bpm = value*monthly_avg_bpm
        print("ExpectedReturn "+key + ": " + str(exp_ret_bpm))
    elif key == "intesa":
        exp_ret_int = value*monthly_avg_int  
        print("ExpectedReturn "+key + ": " + str(exp_ret_int))
        
expected_return_portafoglio = exp_ret_uni + exp_ret_bpm + exp_ret_int
expected_return_portafoglio

#come si può osservare dall'output il valore atteso del portafoglio
#vari a seconda del tipo di trading si voglia attuare. 
#expected return monthly: 0.06903877688101037
#expected return daily: 0.11497798332374587



