"""
Code for assignment one research Implementation
Contains 3 seperate commits now
"""
# import required libraries for the assignment
import sqlite3
from math import *
import numpy as np

# this function connected to the database and extracts all the data as the following variables
# and returns longValL,latValL,productionValL,longValP,latValP lists
def sqlConnection():
    conn = sqlite3.connect('renewable.db')
    c = conn.cursor()
    #get the data from the first table
    cursor = conn.execute("SELECT long,lat,production from location")
    #get the data from table 1
    table1 = cursor.fetchall()
    longValL=[]
    latValL=[]
    productionValL=[]
    for x in range(0, 10):
        longValL.append(table1[x][0])
        latValL.append(table1[x][1])
        productionValL.append(table1[x][2])
     
    #get the data from the second table
    cursor = conn.execute("SELECT long,lat from ports")
    #get the data from table 2
    table2 = cursor.fetchall()
    longValP=[]
    latValP=[]
    for x in range(0, 3):
        longValP.append(table2[x][0])
        latValP.append(table2[x][1])
     
    print "Operation done successfully";
    #print to check values are correct
    #print longValL,latValL,productionValL,longValP,latValP
    #return out of function all the required values
    return longValL,latValL,productionValL,longValP,latValP
    #close the database connection    
    conn.close()
    
    

# now optimise the data for the best solution
# assume flat surface due to short distance in ireland and use simple 
# geometry dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 ) to calculate the distance between the production plants
# use distance times production volume as the value to minimise in the optimisation 
#algorithm
def optimiseDist(longValL,latValL,productionValL,longValP,latValP):
    ## 1st get distance for locataion v location for best distance inlcuding production quantites to be trasnported (72 values/8 each location)   
    productionDistance=[]
    productionDistance1=[]
    for i in range(0, len(longValL)):
        for j in range(0, len(latValL)):
            if i != j:
                #calculate the distance and factor in the production volume                
                dist = sqrt( (latValL[j] - latValL[i])**2 + (longValL[j] - longValL[i])**2)*productionValL[j]
                #store the measure to the list productionDistance with long lat values                 
                productionDistance.append([longValL[i],latValL[i],longValL[j],latValL[j],dist])
    # 2nd  get this total summed value for each location      
    locationTotal=[]
    data=np.asarray(productionDistance)
    for i in range(0, len(data),9):
        locationTotal.append([data[i,0],data[i,1],sum(data[i:i+9,4])])
    
    
    data=np.asarray(locationTotal)
    dataProduction=np.asarray(productionValL)
    totalDataProduction=sum(dataProduction)
    #3rd pass this information to the next phase of the routine which calculates the 27 potential overall values to be be minimised     
    overallResultsMatrix=[]
    for i in range(0, len(data)):
        for j in range(0, len(latValP)):
                dist = sqrt( (latValL[j] - data[i,1])**2 + (longValL[j] - data[i,0])**2)*totalDataProduction+data[i,2]
                overallResultsMatrix.append([longValL[i],latValL[i],longValP[j],latValP[j],dist])
    
    testPrint=np.asarray(overallResultsMatrix) 
    # print check for length check     
    print len(testPrint[:,0])
    
    return productionDistance1, overallResultsMatrix, locationTotal    

  

# main code file 
#get the data from the database 2 tables and get the 5 sets of data required, use lists
#this database information will be passed into the optimisation function
[longValL,latValL,productionValL,longValP,latValP]=sqlConnection() 
 #run the optimisation function
# 1st get distance for locataion v location for best distance inlcuding production quantites to be trasnported (72 values/8 each location)
# 2nd  get this total summed value for each location   
# 3rd pass this information to the next phase of the routine which calculates the 27 potential overall values to be be minimised 
[productionDistance1, overallResultsMatrix, locationTotal] =productionDistance=optimiseDist(longValL,latValL,productionValL,longValP,latValP)
productionDistance1=np.asarray(productionDistance1)
locationTotal=np.asarray(locationTotal)
overallResultsMatrix=np.asarray(overallResultsMatrix)
# print all the relevent output data for the text file report and final checks
print productionDistance1
print ""
print locationTotal
print ""
print overallResultsMatrix
print ""
#get the final Result which is the lowest value
minValue = min(overallResultsMatrix[:,4]) 
minIndex=np.argmin(overallResultsMatrix[:,4])
bestResult=[overallResultsMatrix[minIndex,:]]
#print ""
print bestResult




