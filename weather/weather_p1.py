                                                    # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import tensorflow as tf
import numpy as np


#preprocessing LSTM

dataset=pd.read_csv("final_data.csv")

dataset=dataset.iloc[::-1]
dataset=dataset.reset_index()
dataset=dataset.iloc[:,1:]

list1=[]
for i in range(len(dataset.iloc[:,3])):
    j=dataset.iloc[i,3].split()
    k=dataset.iloc[i,3].split()
    k=k[1].replace("km/h","")
    list1.append(k)
    dataset.iloc[i,3]=float(j[0])
    
list1=pd.DataFrame(list1)
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
list1=enc.fit_transform(np.reshape(list1,(-1,1))).toarray()
list1=list1[:,1:]
list1=pd.DataFrame(list1)

data_p=dataset.iloc[:,:4]
data_p=pd.concat([data_p,list1],axis=1)
data_p=pd.concat([data_p,dataset.iloc[:,4:]],axis=1)


train_dataset=data_p.values



x_train_rnn=[]
y_train_rnn=[]


for i in range(120,3943-120):
    x_train_rnn.append(train_dataset[i-113:i+7,:])
    y_train_rnn.append(train_dataset[i-120:i-113,:])
    
x_train_rnn=np.array(x_train_rnn)
y_train_rnn=np.array(y_train_rnn)






#preprocessing SVM

svm_data=[]
for j in range(1,13):
    try:
        for i in range(1,32):
                    l=np.array(pd.read_csv("data-{}-{}.csv".format(i,j)))
                    l=l[::-1]
                    svm_data.append(l)
    except:
        pass
      
  
for i in range(len(svm_data)):
    for j in range(len(svm_data[i])):
        svm_data[i][j][1]=float(svm_data[i][j][1].split()[0])
        
        
for i in range(len(svm_data)):
    for j in range(len(svm_data[i])):
        svm_data[i][j][2]=float(svm_data[i][j][2].split()[0])
        
list2=[]
for i in range(len(svm_data)):
    for j in range(len(svm_data[i])):
        k=svm_data[i][j][3].split()
        l=svm_data[i][j][3].split()
        l=l[1].replace("km/h","")
        list2.append(l)
        svm_data[i][j][3]=float(svm_data[i][j][3].split()[0])
        
        
for i in range(len(svm_data)):
    for j in range(len(svm_data[i])):
        svm_data[i][j][4]=float(svm_data[i][j][4].replace("%",""))
              
for i in range(len(svm_data)):
    for j in range(len(svm_data[i])):
        svm_data[i][j][5]=float(svm_data[i][j][5].replace("%",""))
        

for i in range(len(svm_data)):
    for j in range(len(svm_data[i])):
        svm_data[i][j][6]=float(svm_data[i][j][6].split()[0])
        
for i in range(len(svm_data)):
    for j in range(len(svm_data[i])):
        svm_data[i][j][7]=float(svm_data[i][j][7].split()[0])
        


list2=np.array(list2)
list2=enc.transform(np.reshape(list2,(-1,1))).toarray()
list2=list2[:,1:]

svm_data_processed=np.array(svm_data[0].astype("float32"))
svm_data_processed=svm_data_processed.reshape(1,11,8)

for i in range(1,len(svm_data)):
    try:
        lp=np.array(svm_data[i].astype("float32"))
        lp=lp.reshape(1,11,8)
        svm_data_processed=np.append(svm_data_processed,lp,axis=0)
    except:
        pass
    
svm_data_processed_2=np.array(svm_data[293].astype("float32"))
svm_data_processed_2=svm_data_processed_2.reshape(1,10,8)


for j in range(294,len(svm_data)):
    try:
        lp=np.array(svm_data[i].astype("float32"))
        lp=lp.reshape(1,10,8)
        svm_data_processed_2=np.append(svm_data_processed_2,lp,axis=0)
    except:
        pass
    
    
    
svm_data_processed=svm_data_processed.reshape(293*11,8)
svm_data_processed_2=svm_data_processed_2.reshape(72*10,8)
svm_data_with_direction=np.append(svm_data_processed,svm_data_processed_2,axis=0)
svm_data_with_direction=np.append(svm_data_with_direction,list2,axis=1)

#normalising 
from sklearn.preprocessing import StandardScaler
sc_1=StandardScaler()
svm_data_with_direction=sc_1.fit_transform(svm_data_with_direction)


svm_data_processed=svm_data_with_direction[0:293*11,:].reshape(293,11,22)
svm_data_processed_2=svm_data_with_direction[293*11:,:].reshape(72,10,22)



x_train_1_p1=svm_data_processed[:,2:,1:]
x_train_1_p2=svm_data_processed[:,1:-1,1:]
y_train_1_p1=svm_data_processed[:,1,1:].reshape(293,1,21)
y_train_1_p2=svm_data_processed[:,0,1:].reshape(293,1,21)

x_train_2_p1=svm_data_processed_2[:,2:,1:]
x_train_2_p2=svm_data_processed_2[:,1:-1,1:]
y_train_2_p1=svm_data_processed_2[:,1,1:].reshape(72,1,21)
y_train_2_p2=svm_data_processed_2[:,0,1:].reshape(72,1,21)



#lstm year wise
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout,BatchNormalization
initializer=tf.keras.initializers.GlorotNormal


model1=Sequential()
model1.add(Dense(128,activation='relu',input_shape=(x_train_1_p1.shape[1],x_train_1_p1.shape[2]),kernel_initializer=initializer))
model1.add(Dropout(0.1))
model1.add(LSTM(512,kernel_initializer=initializer,dropout=0.15,recurrent_dropout=0.15,return_sequences=True,go_backwards=True))
model1.add(BatchNormalization())
#model1.add(LSTM(1024,kernel_initializer=initializer,dropout=0.15,recurrent_dropout=0.15,return_sequences=True,go_backwards=True))
#model1.add(BatchNormalization())
model1.add(LSTM(512,kernel_initializer=initializer,dropout=0.15,recurrent_dropout=0.15,go_backwards=True))
model1.add(BatchNormalization())
#model1.add(Dense(1024,activation='relu',kernel_initializer=initializer))  
#model1.add(Dropout(0.1))
#model1.add(BatchNormalization())
#model1.add(Dense(512,activation='relu',kernel_initializer=initializer))
#model1.add(Dropout(0.1))
#model1.add(BatchNormalization())
model1.add(Dense(21))
#model1.add(BatchNormalization())

model1.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
model1.fit(x_train_1_p1,y_train_1_p1.reshape(293,21),epochs=5,batch_size=128)


model2=Sequential()
model2.add(Dense(128,activation='relu',input_shape=(x_train_2_p1.shape[1],x_train_2_p1.shape[2]),kernel_initializer=initializer))
model2.add(Dropout(0.1))
model2.add(LSTM(512,kernel_initializer=initializer,dropout=0.15,recurrent_dropout=0.15,return_sequences=True,go_backwards=True))
model2.add(BatchNormalization())
#model2.add(LSTM(1024,kernel_initializer=initializer,dropout=0.15,recurrent_dropout=0.15,return_sequences=True,go_backwards=True))
#model2.add(BatchNormalization())
model2.add(LSTM(512,kernel_initializer=initializer,dropout=0.15,recurrent_dropout=0.15,go_backwards=True))
model2.add(BatchNormalization())
#model2.add(Dense(1024,activation='relu',kernel_initializer=initializer))  
#model2.add(Dropout(0.1))
#model2.add(BatchNormalization())
#model2.add(Dense(512,activation='relu',kernel_initializer=initializer))
#model2.add(Dropout(0.1))
#model2.add(BatchNormalization())
model2.add(Dense(21))
#model2.add(BatchNormalization())

model2.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])
model2.fit(x_train_2_p1,y_train_2_p1.reshape(72,21),epochs=5,batch_size=128)


#for loop and apply svm and append the results into list from p1
from sklearn.ensemble import RandomForestRegressor
model_rf=RandomForestRegressor(n_estimators=2000)
#model_rf.fit(x_train_2_p1[0].T,y_train_2_p1[0].T)
model_rf.fit(x_train_1_p1[0].T,y_train_1_p1[0].T)#doubt
model_rf.predict(svm_data_processed[0][1:-1,1:].T)#doubt (Gives better results for some)
#model_rf.predict(x_train_2_p1[1].T)


import csv

with open('filename', 'w') as myfile:
    w = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    w.writerow(svm_data)