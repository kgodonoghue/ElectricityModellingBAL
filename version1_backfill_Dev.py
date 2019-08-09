import pymysql
pymysql.install_as_MySQLdb()
import mysql.connector
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from random import randint
import os
from numpy import genfromtxt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from itertools import product
from math import *
import csv
from sklearn import svm, datasets, preprocessing, tree
from pandas import DataFrame, read_csv
from sklearn.neural_network import MLPClassifier, MLPRegressor
import glob
import pickle
from math import floor
from sklearn import metrics
import random
import glob
import time
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from random import randint
import os
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import sklearn
import statsmodels.api as sm
from sklearn import linear_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from pandas import concat
import datetime
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import mysql.connector
from sqlalchemy import create_engine
from pandas.io import sql
import MySQLdb



''' ML Linear Predictive Function '''
def ML_linear(inputData, outputData):
    clf = linear_model.LinearRegression()
    clf.fit(inputData,outputData)
    predicted_value=clf.predict(inputData)
    return predicted_value,clf

''' ML NN Predictive Function '''
def ML_NN(inputData, outputData, size, alpha_value):
    clf = MLPRegressor(hidden_layer_sizes=(size,),  activation='relu', solver='adam',    alpha=alpha_value,batch_size='auto', learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,  nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,  epsilon=1e-08)
    clf.fit(inputData,outputData)
    predicted_value=clf.predict(inputData)
    return predicted_value,clf

''' ML SMN Predictive Function '''
def ML_Optimizer_NN(xtrain, ytrain,xtest, ytest,range_size,range_alpha):
    results_fit_array=[]
    for i in range(1,range_size):
        for j in range(1,range_alpha):
            network_size=i*10
            alpha_value=j/50 
            [pred_train,clf]=ML_NN(xtrain, ytrain,network_size,alpha_value)
            pred_test=clf.predict(xtest)
            results_train=ytrain-pred_train
            results_test=ytest-pred_test
            results_fit_array.append([network_size,alpha_value,clf,sum(abs(results_train)),sum(abs(results_test))])
            print(network_size,alpha_value,sum(abs(results_train)),sum(abs(results_test)))
    results_fit_array=np.array(results_fit_array)
    lowest_index=np.argmin(results_fit_array[:,3], axis=0)
    print(results_fit_array)
    optimal_size=results_fit_array[lowest_index,0]
    optimal_alpha=results_fit_array[lowest_index,1]
    clf=results_fit_array[lowest_index,2]
    
    return pred_train,optimal_size,optimal_alpha,clf

''' train and test split function '''
def splitPreProcess(input_values,output_values,test_window,lstm_history):
    
    xtrain=input_values[0:len(input_values)-test_window,:]
    ytrain=output_values[0:len(input_values)-test_window]

    xtest=input_values[len(input_values)-test_window-lstm_history:,:]
    ytest=output_values[len(output_values)-test_window-lstm_history:]
    
    return xtrain,xtest,ytrain,ytest

def create_final_input_output(df_final,model_type):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    input_values_forecast=df_final[labels_forecast]
    input_values_historical_short=df_final[labels_historical_short]
    input_values_historical_long=df_final[labels_historical_long]
    output_values=df_final[['smp_d_plus_4','smp_d_minus_1']]
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
 
def create3D(xtrain,xtest,ytrain,ytest,lstm_history):
    
    # create the 3D train
    col_size_train=len(xtrain[0,:])
    row_size_train=lstm_history
    mat_size_train=len(xtrain)-lstm_history
    xtrain_3D=np.zeros((mat_size_train,row_size_train,col_size_train))
    MATindex=0
           
    for i in range(lstm_history,len(xtrain)): 
            xtrain_3D[MATindex,0:row_size_train,0:col_size_train]=xtrain[i-lstm_history:i,:]
            MATindex=MATindex+1
    
    ytrain_3D=ytrain[lstm_history:]
    
    # create the 3D test
    col_size_test=len(xtest[0,:])
    row_size_test=lstm_history
    mat_size_test=len(xtest)-lstm_history
    xtest_3D=np.zeros((mat_size_test,row_size_test,col_size_test))
    MATindex=0
           
    for i in range(lstm_history,len(xtest)): 
            xtest_3D[MATindex,0:row_size_test,0:col_size_test]=xtest[i-lstm_history:i,:]
            MATindex=MATindex+1
    
    ytest_3D=ytest[lstm_history:]
       
    return xtrain_3D,xtest_3D,ytrain_3D,ytest_3D

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
    
    #trip table
    df_trip=pd.read_sql('SELECT * FROM pub_d_dispatchinstructions where INSTRUCTION_CODE = "TRIP"', cnx)
    df_trip=df_trip[['INSTRUCTION_TIMESTAMP','INSTRUCTION_ISSUE_TIME','INSTRUCTION_CODE','RESOURCE_NAME']]
    df_trip['year']=df_trip['INSTRUCTION_ISSUE_TIME'].astype(str).str[0:4]
    df_trip['month']=df_trip['INSTRUCTION_ISSUE_TIME'].astype(str).str[5:7]
    df_trip['day']=df_trip['INSTRUCTION_ISSUE_TIME'].astype(str).str[8:10]
    df_trip['hour']=df_trip['INSTRUCTION_ISSUE_TIME'].astype(str).str[11:13]
    df_trip['RESOURCE_NAME']=df_trip['RESOURCE_NAME'].astype(str).str[3:9]
    df_trip['RESOURCE_NAME']=df_trip['RESOURCE_NAME'].astype(int)
    df_trip['merge_variable'] = df_trip['day'].astype(str) + df_trip['month'].astype(str) + df_trip['year'].astype(str) + df_trip['hour'].astype(str) 
  
    df_smp = df_smp.merge(df_trip,  how='left', on=['merge_variable'])
    #df_smp = df_smp.drop(columns=['year','month','day','hour_y'])
    df_smp = df_smp.drop(columns=['hour_y'])
    
    df_smp['INSTRUCTION_CODE']= df_smp['INSTRUCTION_CODE'].str.replace("TRIP", "1")
    df_smp['INSTRUCTION_CODE']=df_smp['INSTRUCTION_CODE'].fillna(0)
    df_smp['INSTRUCTION_CODE']=df_smp['INSTRUCTION_CODE'].astype(int)
    
    df_smp.rename(columns={'hour_x': 'hour', 'oldName2': 'newName2'}, inplace=True)
    
    for i in range(0,len(df_smp)-10):
        if df_smp['INSTRUCTION_CODE'].iloc[i]==1:
            #df_merged['INSTRUCTION_CODE'].iloc[i:i+4]=2
            df_smp['INSTRUCTION_CODE'].iloc[i:i+4]=2
        else:
            x=0 
    
    df_smp=df_smp.drop_duplicates('unix_date')
     
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
    df_final['hour']= df_final['hour'].str.replace("24", "00")
    df_final['hour']=df_final['hour'].astype(int)
    df_final['time_stamp']=1

    index_time = df_final[df_final['unix_date']==dt].index.values.astype(int)
    index_time = int(index_time)
    print(index_time)
    
    df_final['weekday'] = pd.to_datetime(df_final['unix_date'],unit='s')
    df_final['weekday']=df_final['weekday'].dt.dayofweek
    
    df_final['Bank Holiday']=df_final['dates_x'].isin([1012019,170319,18032019,22042019,6052019,3062019,5082019,28102019,25122019,26122019])
    df_final['Bank Holiday']=df_final['Bank Holiday'].astype(int)
 
   
    df_final=df_final.iloc[::sample_interval, :]
   
    df_final=df_final.iloc[1000:index_time]
    df_final=df_final.fillna(0) 
    df_final.to_csv('final_file.csv',index=False,header=True)
    df_final = pd.read_csv('final_file.csv')
   
    
    labels_forecast=['smp_d_minus_1','hour_interval', 'WindForecastEirgrid','hour','weekday','Bank Holiday','TSODemandForecast','TSORenewableForecast']
    labels_historical_short=['smp_d_plus_4','smp_d_minus_1','TSODemandForecast','TSORenewableForecast','INSTRUCTION_CODE','sum_power','pressure_value','temperature_value','sum_power','smp_d_plus_4','smp_d_minus_1','FUEL_GAS','FUEL_OTHER_FOSSIL', 
                            'co2','FUEL_COAL','hour','MIX_COAL', 'TotalPN', 'DemandActualEirgrid', 'WindActual', 
                            'FUEL_RENEW','GBP', 'TSODemandForecast','TSORenewableForecast','INSTRUCTION_CODE','gas']
    labels_historical_long=['sum_power','pressure_value','temperature_value','sum_power','smp_d_plus_4','smp_d_minus_1','FUEL_GAS','FUEL_OTHER_FOSSIL', 
                            'co2','FUEL_COAL','hour','MIX_COAL', 'TotalPN', 'DemandActualEirgrid', 'WindActual', 
                            'FUEL_RENEW','GBP', 'TSODemandForecast','TSORenewableForecast','INSTRUCTION_CODE','gas']
    labels_historical_long=labels_historical_long+list(power_stations_only)
    all_labels=labels_forecast+labels_historical_short+labels_historical_long
    
    return all_labels,df_final,labels_forecast,labels_historical_short,labels_historical_long

def create_model_output(model_type,input_values_combined, output_values_range,count,clf,no_epochs):
        if model_type=='regression':
            ''' ML regression Model '''
            [xtrain,xtest,ytrain,ytest]=splitPreProcess(input_values_combined,output_values_range,test_window,lstm_history)
            [pred_train,clf]=ML_linear(xtrain, ytrain)
            pred_train=clf.predict(xtrain)
            pred_test=clf.predict(xtest[lstm_history:,:])
            pred_train= pred_train.ravel()
            pred_test= pred_test.ravel()
            ytest=ytest[lstm_history:]
        elif model_type=='neural net':
            ''' NN regression Model '''
            [xtrain,xtest,ytrain,ytest]=splitPreProcess(input_values_combined,output_values_range,test_window,lstm_history)
            [pred_train,optimal_size,optimal_alpha,clf]=ML_Optimizer_NN(xtrain, ytrain,xtest, ytest,range_size,range_alpha)
            pred_train=clf.predict(xtrain)
            pred_test=clf.predict(xtest[lstm_history:,:])
            pred_train= pred_train.ravel()
            pred_test= pred_test.ravel()
            ytest=ytest[lstm_history:]
        elif model_type=='RF':
            [xtrain,xtest,ytrain,ytest]=splitPreProcess(input_values_combined,output_values_range,test_window,lstm_history)
            clf = RandomForestRegressor(max_depth=5, random_state=0,n_estimators=10)
            clf.fit(xtrain, ytrain)
            print(clf.feature_importances_)
            pred_train=clf.predict(xtrain)
            pred_test=clf.predict(xtest[lstm_history:,:])
            pred_train= pred_train.ravel()
            pred_test= pred_test.ravel()
            ytest=ytest[lstm_history:]
        elif model_type=='lstm':
            ''' LSTM regression Model '''    
            [xtrain,xtest,ytrain,ytest]=splitPreProcess(input_values_combined,output_values_range,test_window,lstm_history)
            xtrain = xtrain.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
            xtest = xtest.reshape((xtest.shape[0], 1, xtest.shape[1]))
            if count==0:
                clf = Sequential()
                clf.add(LSTM(200, input_shape=(xtrain.shape[1], xtrain.shape[2])))
                clf.add(Dense(1))
                clf.compile(loss='mae', optimizer='adam')
                history = clf.fit(xtrain, ytrain, epochs=10, batch_size=5, validation_data=(xtrain, ytrain), verbose=1, shuffle=True)
                pyplot.plot(history.history['loss'], label='train')
                pyplot.plot(history.history['val_loss'], label='test')
                pyplot.legend()
                pyplot.show()
                pred_train=clf.predict(xtrain)
                pred_test=clf.predict(xtest[lstm_history:,:])
                pred_train= pred_train.ravel()
                pred_test= pred_test.ravel()
                ytest=ytest[lstm_history:]
                s = pickle.dumps(clf)
            else:
                clf = pickle.loads(s)
                pred_train=clf.predict(xtrain)
                pred_test=clf.predict(xtest[lstm_history:,:])
                pred_train= pred_train.ravel()
                pred_test= pred_test.ravel()
                ytest=ytest[lstm_history:]
        elif model_type=='lstm_3D':
            ''' LSTM regression Model '''    
            [xtrain,xtest,ytrain,ytest]=splitPreProcess(input_values_combined,output_values_range,test_window,lstm_history)
            [xtrain_3D,xtest_3D,ytrain_3D,ytest_3D]=create3D(xtrain,xtest,ytrain,ytest,lstm_history)
            xtrain=xtrain_3D
            xtest=xtest_3D
            ytrain=ytrain_3D
            ytest=ytest_3D            
            if (count==0) or (count>0):
                clf = Sequential() 
                clf.add(LSTM(200, input_shape=(xtrain.shape[1], xtrain.shape[2])))
                clf.add(Dense(1))
                clf.compile(loss='mae', optimizer='adam')
                history = clf.fit(xtrain, ytrain, epochs=no_epochs, batch_size=10, validation_data=(xtrain, ytrain), verbose=1, shuffle=True)
                pyplot.plot(history.history['loss'], label='train')
                pyplot.plot(history.history['val_loss'], label='test')
                pyplot.legend()
                pyplot.show()
                pred_train=clf.predict(xtrain)  
                pred_test=clf.predict(xtest)
                pred_train= pred_train.ravel()
                pred_test= pred_test.ravel()
                s = pickle.dumps(clf)
            else:
                #clf = pickle.loads(s)
                pred_train=clf.predict(xtrain)  
                pred_test=clf.predict(xtest)
                pred_train= pred_train.ravel()
                pred_test= pred_test.ravel()
        
        corr_coefficient_train=np.corrcoef(pred_train,ytrain)
        corr_coefficient_test=np.corrcoef(pred_test,ytest)
        final_performance.append([model_type,np.sum(abs(pred_test-ytest)),corr_coefficient_test[0,1],np.sum(abs(pred_train-ytrain)),corr_coefficient_train[0,1]])
        print(final_performance)
        df_performance=pd.DataFrame(final_performance)
        df_performance.to_csv('df_performance.csv',index=False,header=True)
        return pred_train,pred_test,ytrain,ytest,clf


path_dukascopy =r'C:/main_folder/sql_modelling/dukascopy'
root_path = r'C:/main_folder/sql_modelling'
user_name = 'fergus'
passw = 'Uniwhite_8080'
host_IP =  '185.176.0.173'
port = 3306
database_name = 'smartpow_world'
corr_limit=0   
forward_look_short=48
forward_look_long=96
test_window=48 
index_start_input_long=2000
index_start_output=index_start_input_long+forward_look_long 
index_start_input_short=index_start_input_long+forward_look_short 
range_size=5
range_alpha=5
start_smp_database=24000
stop_smp_database=40000
window_smp_database=15000
model_list=['regression'] 
model_list=['lstm_3D']
model_list=['RF']
final_performance=[]
lstm_history=96
count=0
clf=0
interval_delay=0
no_epochs=20
number_days=40
time_interval_test=86400
sample_interval=1

if __name__ == '__main__':
    
    import time
    dt = datetime.datetime(2019, 6, 1 , 20 , 00 ) 
    dt=time.mktime(dt.timetuple())
    for i in range(0,number_days,1):
        for model_type in model_list:
            # create the input and output variables for modelling
            start_time = time.time()
            [all_labels,df_final,labels_forecast,labels_historical_short,labels_historical_long]=createDF(user_name, passw, host_IP, database_name,dt)
            [input_values_combined, output_values_range]=create_final_input_output(df_final,model_type)
            [pred_train,pred_test,ytrain,ytest,clf]=create_model_output(model_type,input_values_combined, output_values_range,count,clf,no_epochs)        
            data = pd.DataFrame()
            df_output = pd.DataFrame(data, columns = ['unix_date','Date', 'Hour', 'hour_interval', 'Model_BAL','Actual_BAL']) 
            df_output['Date']=df_final['dates_x'].iloc[len(df_final)-test_window:]
            df_output['Hour']=df_final['hour'].iloc[len(df_final)-test_window:]
            df_output['hour_interval']=df_final['hour_interval'].iloc[len(df_final)-test_window:]
            df_output['unix_date']=df_final['unix_date'].iloc[len(df_final)-test_window:]
           
            min_max_scaler = preprocessing.MinMaxScaler()
            output_values=df_final[['smp_d_plus_4','smp_d_minus_1']]
            output_values = min_max_scaler.fit_transform(output_values)
            output_values_descaled = min_max_scaler.inverse_transform(output_values)
            pred_test_reshape=np.tile(pred_test.reshape(len(pred_test), 1), (1, 2))
            pred_test_descaled = min_max_scaler.inverse_transform(pred_test_reshape)
            df_output['Model_BAL']=pred_test_descaled[:,0]
           
            ytest_reshape=np.tile(ytest.reshape(len(ytest), 1), (1, 2))
            ytest_descaled = min_max_scaler.inverse_transform(ytest_reshape)
            df_output['Actual_BAL']=df_final['smp_d_plus_4'].iloc[len(df_final)-test_window:]
            df_output['Actual_DAM']=df_final['smp_d_minus_1'].iloc[len(df_final)-test_window:]
            df_output.to_csv('df_output.csv',index=False,header=True)

            if count==0:
                engine = create_engine('mysql+mysqldb://fergus:Uniwhite_8080@185.176.0.173:3306/smartpow_world', echo = False)
                df_output.to_sql(name='Forecast_BAL_Dev', con=engine, if_exists = 'replace', index=False)
            if count>0:
                engine = create_engine('mysql+mysqldb://fergus:Uniwhite_8080@185.176.0.173:3306/smartpow_world', echo = False)
                df_output.to_sql(name='Forecast_BAL_Dev', con=engine, if_exists = 'append', index=False)
         
            count=count+1
            dt=dt+time_interval_test
   
 

   
 
