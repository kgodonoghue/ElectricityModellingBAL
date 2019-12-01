# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:26:16 2019

@author: kgodo
"""

'''IDA1 closes at 5:30 and goes from 11pm tonight to 11pm tomorrow night. 

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
    df_smp = pd.read_sql('SELECT * FROM duos_tuos_semo_smp_backup limit 100000,110000', cnx)
    df_smp=df_smp.iloc[start_smp_database:stop_smp_database]
    #window_smp_database = df_smp[df_smp['unix_date']==dt].index.values.astype(int)
    df_smp=df_smp.iloc[0:window_smp_database]
    df_smp['GBP']=df_smp['GBP_DAM']/df_smp['smp_d_minus_1']
    df_smp['hour']=df_smp['hour'].astype(str)
    df_smp['hour'] = df_smp['hour'].apply(lambda x: x.zfill(2))
    df_smp['dates']=df_smp['dates'].astype(str)
    df_smp['dates'] = df_smp['dates'].str.replace("/", "")
    df_smp['merge_variable'] = df_smp['dates'].astype(str) + df_smp['hour'].astype(str) 
    
    print('smp')
    
     
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
    
    print('weather')
    
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
   
    df_final=df_final.iloc[1000:index_time]
    df_final=df_final.fillna(0) 
    df_final.to_csv('final_file.csv',index=False,header=True)
    df_final = pd.read_csv('final_file.csv')
    
    #df_final=df_final.groupby(df_final.index // N).mean()
    
    df_final['DAM BAL Delta']=df_final['smp_d_minus_1']-df_final['smp_d_plus_4']
    df_final['DAM BAL Delta']=df_final['DAM BAL Delta'].apply(lambda x:0 if x<=0 else 1)
   
    labels_forecast=['smp_d_minus_1','WindForecastEirgrid','hour','weekday']
    labels_historical_short=['smp_d_plus_4','smp_d_minus_1','gas', 'temperature_value', 'pressure_value', 'humidity']
    labels_historical_long=['smp_d_minus_1','gas']
    #labels_historical_short=labels_historical_short+list(power_stations_only)
    
    print('done')
    
    all_labels=labels_forecast+labels_historical_short+labels_historical_long
    
    return all_labels,df_final,labels_forecast,labels_historical_short,labels_historical_long

    
def create_model_output(model_type,input_values_combined, output_values_range):

        if model_type=='RF':
            [xtrain,xtest,ytrain,ytest]=splitPreProcess(input_values_combined,output_values_range,test_window)
            clf = RandomForestRegressor(max_depth=4, random_state=0,n_estimators=5)
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
forward_look_short=100
forward_look_long=200
index_start_input_long=1000
index_start_output=index_start_input_long+forward_look_long 
index_start_input_short=index_start_input_long+forward_look_short 
start_smp_database=24000
stop_smp_database=40000
window_smp_database=15000
model_list=['RF']
number_days=150 
sample_interval=1
back_test_file=0
wait_delay=1800 
count=1
test_window=96
threshold=10
next_day_delay=86400
window=7200

if __name__ == '__main__':
    
    import time
    dt = datetime.datetime(2019, 12, 2  , 23 , 00 ) 
    dt=time.mktime(dt.timetuple())
    start_time = time.time()
    [all_labels,df_final,labels_forecast,labels_historical_short,labels_historical_long]=createDF(user_name, passw, host_IP, database_name,dt)
    df_final['DAM BAL Delta']=df_final['smp_d_minus_1']-df_final['smp_d_plus_4']
    
    for i in range(0,number_days,1):
        for model_type in model_list:
            df_final1=df_final.iloc[0:len(df_final)-window]
            window=window-48
            [input_values_combined, output_values_range]=create_final_input_output(df_final1,model_type)
            [pred_train,pred_test,ytrain,ytest,clf]=create_model_output(model_type,input_values_combined, output_values_range)        
       
            data = pd.DataFrame()
            df_output = pd.DataFrame(data, columns = ['unix_date','Date', 'Hour', 'hour_interval', 'Model_BAL','Actual_BAL']) 
            df_output['Date']=df_final1['dates_x'].iloc[len(df_final1)-test_window:]
            df_output['Hour']=df_final1['hour'].iloc[len(df_final1)-test_window:]
            df_output['hour_interval']=df_final1['hour_interval'].iloc[len(df_final1)-test_window:]
            df_output['unix_date']=df_final1['unix_date'].iloc[len(df_final1)-test_window:]
            min_max_scaler = preprocessing.MinMaxScaler()
            output_values=df_final1[['DAM BAL Delta','smp_d_minus_1']]
            output_values = min_max_scaler.fit_transform(output_values)
            output_values_descaled = min_max_scaler.inverse_transform(output_values)
            pred_test_reshape=np.tile(pred_test.reshape(len(pred_test), 1), (1, 2))
            pred_test_descaled = min_max_scaler.inverse_transform(pred_test_reshape)
            df_output['Predicted_Delta']=pred_test_descaled[:,0]
            df_output['Model_BAL']=df_output['Predicted_Delta']
            df_output['Predicted_Delta']=df_output['Predicted_Delta'].apply(lambda x:0 if x<=threshold else 1)           
            df_output['Actual_BAL']=df_final1['smp_d_plus_4'].iloc[len(df_final1)-test_window:]
            df_output['Actual_DAM']=df_final1['smp_d_minus_1'].iloc[len(df_final1)-test_window:]
            df_output['Actual_Delta']=df_final1['DAM BAL Delta'].iloc[len(df_final1)-test_window:]
            df_output['Actual_Delta_Binary']=df_output['Actual_Delta'].apply(lambda x:0 if x<=0 else 1)
            df_DAM_prediction=df_output[df_output['Predicted_Delta']==1] 
            #df_DAM_prediction['Actual_Delta'].cumsum().plot()
            df_output['WindForecastEirgrid']=df_final1['WindForecastEirgrid'].iloc[len(df_final1)-test_window:]
            sum_power=df_final1['sum_power'].iloc[len(df_final1)-test_window-forward_look_short:-forward_look_short]
            sum_power=np.array(sum_power)
            df_output['sum_power']=sum_power
            df_output['Date'] = df_output['Date'].astype(str)   
            df_output['Date'] = df_output['Date'].apply(lambda x: x.zfill(8))
            df_output['Date'] = df_output['Date'].str[0:2]+'/'+df_output['Date'].str[2:4]+'/'+df_output['Date'].str[4:8]
            df_output =df_output.drop(df_output.index[0:-48])
            df_output.to_csv('df_output.csv',index=False,header=True)

            #df_DAM_prediction['Actual_Delta'].cumsum().plot()

            if count==0:
                engine = create_engine('mysql+mysqldb://fergus:Uniwhite_8080@185.176.0.173:3306/smartpow_world', echo = False)
                df_output.to_sql(name='Forecast_BAL_Dev_IDA1', con=engine, if_exists = 'replace', index=False)
            if count>0:
                engine = create_engine('mysql+mysqldb://fergus:Uniwhite_8080@185.176.0.173:3306/smartpow_world', echo = False)
                df_output.to_sql(name='Forecast_BAL_Dev_IDA1', con=engine, if_exists = 'append', index=False)
            
            dt=dt+next_day_delay
            
            for j in range(0,10000000):
                time.sleep(5)
                stop_time = time.time()
                print(stop_time-start_time)
                #if stop_time-start_time>next_day_delay:
                if stop_time-start_time>0:
                    break 
                
            count=count+1
            
            if model_type=='RF':
                df_feature = pd.DataFrame(columns=['Feature', 'Importance Weight'])
                df_feature['Importance Weight']=clf.feature_importances_
                df_feature['Feature']=all_labels
                df_feature=df_feature.sort_values(by='Importance Weight', ascending=False)
                df_feature.to_csv('feature_list_DAM.csv',index=False,header=True) 
 

            
            

            
            
