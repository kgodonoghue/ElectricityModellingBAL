'''IDA1 closes at 5:30PM and goes from 11pm tonight to 11pm tomorrow night. 

IDA2 closes at 8am  and goes from 11am to 11pm

IDA3  closes at 2:30pm and goes from 5pm to 11 pm (same day).'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from pandas import DataFrame, read_csv
import time
import matplotlib.pyplot as plt
from math import *
from random import randint
from sklearn.preprocessing import MinMaxScaler
import datetime
import time
from pandas_ml import ConfusionMatrix
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

import mysql.connector
from sqlalchemy import create_engine
from pandas.io import sql
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
import mysql.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

''' train and test split function '''
def splitPreProcess(input_values,output_values,test_window):
    
    xtrain=input_values[0:len(input_values)-test_window,:]
    ytrain=output_values[0:len(input_values)-test_window]

    xtest=input_values[len(input_values)-test_window:,:]
    ytest=output_values[len(output_values)-test_window:]
    
    return xtrain,xtest,ytrain,ytest

def create_final_input_output(df_final,model_type):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    input_values_forecast=df_final[labels_forecast]
    input_values_historical_short=df_final[labels_historical_short]
    input_values_historical_long=df_final[labels_historical_long]
    output_values=df_final[['DAM BAL Delta','smp_d_minus_1']]
    output_values = min_max_scaler.fit_transform(output_values)
    #output_values_descaled = min_max_scaler.inverse_transform(output_values)
    output_values=output_values[:,0]
    # set forecast values and csv save
    input_values_forecast = min_max_scaler.fit_transform(input_values_forecast)
    input_values_historical_short = min_max_scaler.fit_transform(input_values_historical_short)
    input_values_historical_long = min_max_scaler.fit_transform(input_values_historical_long)
    ''' end of old forecast variables '''
    ''' create the output and forecasted aligned '''
    output_values_range=output_values[index_start_output:]
    input_values_forecast_range=input_values_forecast[index_start_output:,:]
    ''' set up hist forecast variables '''   
    #input_values_historical_short_range=input_values_historical_short[index_start_input_short:stop_short,:]
    input_values_historical_short_range=input_values_historical_short[index_start_input_short:-forward_look_short]
    ''' set up historical variables '''
    #input_values_historical_long_range=input_values_historical_long[index_start_input_long:stop_long,:]
    input_values_historical_long_range=input_values_historical_long[index_start_input_long:-forward_look_long]
   
    ''' merge the input values '''
    input_values_combined=np.hstack((input_values_forecast_range,input_values_historical_short_range))
    input_values_combined=np.hstack((input_values_combined,input_values_historical_long_range))
    return input_values_combined, output_values_range
 

def createDF(user_name, passw, host_IP, database_name,dt):
    cnx = mysql.connector.connect(user=user_name, password=passw,host=host_IP, database=database_name)
    #smp table
    df_smp = pd.read_sql('SELECT * FROM duos_tuos_semo_smp limit 100000,110000', cnx)
    df_smp=df_smp.iloc[start_smp_database:stop_smp_database]
    #window_smp_database = df_smp[df_smp['unix_date']==dt].index.values.astype(int)
    df_smp=df_smp.iloc[0:window_smp_database]
    df_smp['GBP']=df_smp['GBP_DAM']/df_smp['smp_d_minus_1']
    df_smp['hour']=df_smp['hour'].astype(str)
    df_smp['hour'] = df_smp['hour'].apply(lambda x: x.zfill(2))
    df_smp['dates']=df_smp['dates'].astype(str)
    df_smp['dates'] = df_smp['dates'].str.replace("/", "")
    df_smp['merge_variable'] = df_smp['dates'].astype(str) + df_smp['hour'].astype(str)    
    
     
    #fuel table
    df_fuel=pd.read_sql('SELECT * FROM eirgrid_fuelmix', cnx)
    df_fuel.rename(columns={'systime': 'unix_date'}, inplace=True)
    df_all1=df_smp.merge(df_fuel, how='left', on=['unix_date'])
    df_all1=df_all1.drop_duplicates('unix_date')
    
    #imbalance table
    df_imbalance=pd.read_sql('SELECT * FROM PUB_HrlyForecastImbalance', cnx)
    df_imbalance=df_imbalance.drop_duplicates('unix_date')
    df_final=df_all1.merge(df_imbalance, how='left', on=['unix_date'])
    df_final=df_final.drop_duplicates('unix_date')
    
    #weather table
    df_metweather=pd.read_sql('SELECT * FROM met_weather_half_hour', cnx)
    df_metweather=df_metweather.drop_duplicates('unix_time')
    df_metweather.rename(columns={'unix_time': 'unix_date'}, inplace=True)
    df_metweather[df_metweather['location'] == 'Dublin']
    df_final=df_final.merge(df_metweather, how='left', on=['unix_date']) 
    df_final=df_final.drop_duplicates('unix_date')
    
    #powerstation table
    df_powerstation=pd.read_sql('SELECT * FROM Units_Running', cnx)
    df_powerstation=df_powerstation.drop_duplicates('unix_date')
    df_powerstation=df_powerstation.fillna(0)
    power_stations_only=df_powerstation
    power_stations_only=power_stations_only.drop(columns=['unix_date','dates','id'])
    df_powerstation['sum_power']=power_stations_only.sum(axis=1)
    df_final=df_final.merge(df_powerstation, how='left', on=['unix_date']) 
    df_final=df_final.drop_duplicates('unix_date')

    
    df_final['day']=df_final['dates_x'].astype(str).str[0:2]
    df_final['day']=df_final['day'].astype(int)
    df_final['month']=df_final['dates_x'].astype(str).str[2:4]
    df_final['month']=df_final['month'].astype(int)
    df_final['year']=df_final['dates_x'].astype(str).str[4:8]
    df_final['year']=df_final['year'].astype(int)
    #df_final['hour']= df_final['hour'].str.replace("24", "00")
    df_final['hour']=df_final['hour'].astype(int)
    df_final['time_stamp']=1

    index_time = df_final[df_final['unix_date']==dt].index.values.astype(int)
    index_time = int(index_time)
    print(index_time)
    
    df_final['weekday'] = pd.to_datetime(df_final['unix_date'],unit='s')
    df_final['weekday']=df_final['weekday'].dt.dayofweek
    
    df_final['Bank Holiday']=df_final['dates_x'].isin([1012019,170319,18032019,22042019,6052019,3062019,5082019,28102019,25122019,26122019])
    df_final['Bank Holiday']=df_final['Bank Holiday'].astype(int)
 
    
    #df_final['DAM BAL Delta']=df_final['smp_d_plus_4']
    df_final=df_final.iloc[::sample_interval, :]
   
    df_final=df_final.iloc[0:index_time]
    df_final=df_final.fillna(0) 
    df_final.to_csv('final_file.csv',index=False,header=True)
    df_final = pd.read_csv('final_file.csv')
    
    #df_final=df_final.groupby(df_final.index // N).mean()
    
    df_final['DAM BAL Delta']=df_final['smp_d_minus_1']-df_final['smp_d_plus_4']
    df_final['DAM BAL Delta']=df_final['DAM BAL Delta'].apply(lambda x:0 if x<=0 else 1)
   
    labels_forecast=['TSORenewableForecast' ,'TSODemandForecast','smp_d_minus_1','WindForecastEirgrid','hour','GBP_DAM','NetPosition','IndexVolumes','CalculatedImbalance','NetInterconnectorSchedule','TotalPN','weekday']
    labels_historical_short=['smp_d_minus_1','sum_power','gas','smp_d_plus_4', 'MIX_COAL', 'MIX_GAS', 'MIX_NET_IMPORT', 'MIX_OTHER_FOSSIL', 'MIX_RENEW', 'MIX_TOTAL', 'FUEL_COAL', 'FUEL_GAS', 'FUEL_NET_IMPORT', 'FUEL_OTHER_FOSSIL', 'FUEL_RENEW', 'temperature_value', 'pressure_value', 'humidity']
    labels_historical_long=['smp_d_minus_1','sum_power','gas','smp_d_plus_4']
    #labels_historical_short=labels_historical_short+list(power_stations_only)
    
    all_labels=labels_forecast+labels_historical_short+labels_historical_long
    
    return all_labels,df_final,labels_forecast,labels_historical_short,labels_historical_long

    
def create_model_output(model_type,input_values_combined, output_values_range):

        if model_type=='RF':
            [xtrain,xtest,ytrain,ytest]=splitPreProcess(input_values_combined,output_values_range,test_window)
            clf = RandomForestRegressor(max_depth=5, random_state=0,n_estimators=10)
            clf.fit(xtrain, ytrain)
            #print(clf.feature_importances_)
            pred_train=clf.predict(xtrain)
            pred_test=clf.predict(xtest)
            pred_train= pred_train.ravel()
            pred_test= pred_test.ravel()
        return pred_train,pred_test,ytrain,ytest,clf


path_dukascopy =r'C:/main_folder/sql_modelling/dukascopy'
root_path = r'C:/main_folder/sql_modelling'
user_name = 'fergus'
passw = 'Uniwhite_8080'
host_IP =  '185.176.0.173'
port = 3306
database_name = 'smartpow_world'
move_avg_value=400
result_type='IDA1'
#result_type='DAM'
model='Model_reduced_variable_'
month_start=8
day_start=1
month_end=11
day_end=28
date=str(day_start)+"_"+str(month_start)+"_"+str(day_end)+"_"+str(month_end)
x_spacing=20
move_avg_value=2
DAM_threshold=11
BAL_threshold=-2
cost=0.75

if __name__ == '__main__':
    
    #time and connection
    dt = datetime.datetime(2019, month_start, day_start , 0 , 00) 
    dt=time.mktime(dt.timetuple())
    dt_stop = datetime.datetime(2019, month_end, day_end , 23 , 00) 
    dt_stop = time.mktime(dt_stop.timetuple())    
    cnx = mysql.connector.connect(user=user_name, password=passw,host=host_IP, database=database_name)
    
    #results table
    df_results = pd.read_sql('SELECT * FROM Forecast_BAL_Dev_IDA1', cnx)
    df_results['unix_date']=df_results['unix_date'].astype(int)
    df_results=df_results[df_results['unix_date']>dt]
    df_results=df_results[df_results['unix_date']<dt_stop]
    df_results['Actual_BAL']=df_results['Actual_BAL'].astype(int)
    df_results['Model_BAL']=df_results['Model_BAL'].rolling(move_avg_value).mean()
    df_results['Model_BAL']=df_results['Model_BAL']
    
   
    #df_results=df_results[df_results['IDA1']<1000]
    
    #select which result type IDA 1 or DAM
    if result_type=='DAM':
        df_results['Delta']=df_results['Actual_DAM']-df_results['Actual_BAL']
    elif result_type=='IDA1':
        df_results['Delta']=df_results['IDA1']-df_results['Actual_BAL']
    
    #optional moving average option
    df_results_filter_DAM=df_results[df_results['Model_BAL']>DAM_threshold]
    df_results_filter_DAM['Delta']=df_results_filter_DAM['Delta']-cost
    df_results_filter_BAL=df_results[df_results['Model_BAL']<BAL_threshold]
    
    #df_results_filter_DAM=df_results[df_results['Normal_Trade']==1]
    #df_results_filter_BAL=df_results[df_results['Reverse_Trade']==1]
    
    df_results_filter_BAL['Delta']=-1*df_results_filter_BAL['Delta']-cost
    df_results_filter_BALDAM=df_results_filter_DAM.append(df_results_filter_BAL)  
    df_results_filter_BALDAM=df_results_filter_BALDAM.sort_values(by='unix_date', ascending=True)
    
    fig, ax = plt.subplots(figsize=(20,15))
    ax.grid(linestyle='-', linewidth='0.5', color='grey')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.xticks(rotation = 90)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)
    df_results['Date']=df_results['Date'].astype(str)
    df_results['Date']= df_results['Date'].str.replace("/", "")
    df_results_filter_DAM['Date']=df_results_filter_DAM['Date'].astype(str)
    df_results_filter_DAM['Date']= df_results_filter_DAM['Date'].str.replace("/", "")
    df_results_filter_BAL['Date']=df_results_filter_BAL['Date'].astype(str)
    df_results_filter_BAL['Date']= df_results_filter_BAL['Date'].str.replace("/", "")
    df_results_filter_BALDAM['Date']=df_results_filter_BALDAM['Date'].astype(str)
    df_results_filter_BALDAM['Date']= df_results_filter_BALDAM['Date'].str.replace("/", "")
    plt.xticks(rotation=90)
    plt.plot(df_results['Date'],df_results['Delta'].cumsum(),'b-.',label='Profit Curve 24HR__mean_profit_per_trade__'+ str(df_results['Delta'].mean())  + '_no_trades_' +  str(len(df_results['Delta'])))
    plt.plot(df_results_filter_DAM['Date'],df_results_filter_DAM['Delta'].cumsum(),'r-.',label='Profit Curve Reduced Algo DAM>BAL__mean_profit_per_trade__' + str(df_results_filter_DAM['Delta'].mean()) + '_no_trades_' + str(len(df_results_filter_DAM['Delta'])))
    plt.plot(df_results_filter_BAL['Date'],(df_results_filter_BAL['Delta']).cumsum(),'g-.',label='Profit Curve Reduced Algo BAL>DAM__mean_profit_per_trade__' + str( df_results_filter_BAL['Delta'].mean()) + '_no_trades_' + str(len(df_results_filter_BAL['Delta'])))
    plt.plot(df_results_filter_BALDAM['Date'],(df_results_filter_BALDAM['Delta']).cumsum(),'p-.',label='Profit Curve Reduced Algo Combined__mean_profit_per_trade__' + str(df_results_filter_BALDAM['Delta'].mean()) + '_no_trades_' + str(len(df_results_filter_BALDAM['Delta'])))
    plt.legend(loc='lower left', prop={'size': 15})    
    fig.suptitle('Strategy cumulative difference over time', fontsize=25)
    plt.ylabel('DAM BAL Delta Cumulative Delta (30 minute intervals)', fontsize=25)
    plt.xlabel('Date asscoiated with 30 minute samples', fontsize=25) 
    fig.savefig(model + 'profit_curve.jpg')

    
