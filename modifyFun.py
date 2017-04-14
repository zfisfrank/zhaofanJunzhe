# /usr/local/bin/python3

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# normalize to 0~1, first: dataframe to normalize, 2nd, ignore column name list
def normalize(df, despiteCol = ['userId']):
    interstedCol = df.columns[~df.columns.isin(despiteCol)]
    result = (df[interstedCol] - df[interstedCol].min())/(df[interstedCol].max() - df[interstedCol].min())
    result = pd.concat([df[despiteCol],result],axis = 1)
    # df = (df - df.min())/(df.max() - df.min())
    result = result.dropna(how ='all',axis = 1)
    return result

# normalize to zScore, first: dataframe to normalize, 2nd, ignore column name list
def zScoreNormalize(df, despiteCol = ['userId']):
    interstedCol = df.columns[~df.columns.isin(despiteCol)]
    # result = (df[interstedCol] - df[interstedCol].min())/(df[interstedCol].max() - df[interstedCol].min())
    result = (df[interstedCol] - df[interstedCol].mean())/df[interstedCol].std()
    result = pd.concat([df[despiteCol],result],axis = 1)
    # df = (df - df.min())/(df.max() - df.min())
    result = result.dropna(how ='all',axis = 1)
    return result

# calculate all pearson correlation coefficient of columns in dataFrame with y(target)
def pearCorr(df,y):
    result =  df.apply(lambda x: pearsonr(x,y)).apply(pd.Series)
    result.set_axis(1,['Pearson’s correlation coefficient','2-tailed p-value'])
    return result

from sklearn import metrics
def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()



import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import matplotlib.pyplot as pltdd
import pickle
from sklearn.externals import joblib
def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()
    # read the datasaa

def readMoneyData():
    moneyTotalFlat1 = pd.read_csv('../newFeatures/moneyTotalFlat1.csv')
    moneyTotalFlat1 = moneyTotalFlat1.set_index('userId')
    moneyTotalFlat1 = moneyTotalFlat1.dropna(axis = 1, how = 'all')
    moneyTotalFlat1 = moneyTotalFlat1.dropna(axis = 0, how = 'all')
    moneyTotalFlat2 = pd.read_csv('../newFeatures/moneyTotalFlat2.csv')
    moneyTotalFlat2 = moneyTotalFlat2.set_index('userId')
    moneyTotalFlat2 = moneyTotalFlat2.dropna(axis = 1, how = 'all')
    moneyTotalFlat2 = moneyTotalFlat2.dropna(axis = 0, how = 'all')
    return [moneyTotalFlat1,moneyTotalFlat2]

def readUserInfo():
    userInfo = pd.read_csv('train/user_info_train.txt',names = ['userId','gender','job', 'education', 'marriage', 'residentialType']).set_index('userId')
    loanTime = pd.read_csv('train/loan_time_train.txt',names = ['userId','loanTime']).set_index('userId')
    labels = pd.read_csv('train/overdue_train.txt',names = ['userId','overDueLabel']).set_index('userId')
    totalInfo = pd.concat([userInfo,loanTime],axis = 1)
    return(totalInfo,labels)
def combineAllInfo():
    [moneyTotalFlat1, moneyTotalFlat2] = readMoneyData()
    [fullInfo,labels] = readUserInfo()
    allData = pd.concat([fullInfo,moneyTotalFlat1],axis = 1)
    allData['loanTime'] = allData['loanTime']//2638000
    return [allData,labels]

def dropnaByPrecent(allData,axisVal=0,nonNanPercentile = 0.001): #axis = 0 for drop row, axisVal = 1 for drop columns
    # nonNanPercentile = 0.
    [rowThresh,colThresh] = list(allData.shape)
    rowThresh = np.ceil(rowThresh * nonNanPercentile)
    colThresh = np.ceil(colThresh * nonNanPercentile)
    threshs = [rowThresh,colThresh]
    dropName =[' row',' column']
    print('drop ', dropName[axisVal], ' minimum data', threshs[axisVal])
    allData2 = allData.dropna(axis=axisVal, thresh = threshs[axisVal] )
    print('before drop shape: ',allData.shape)
    print('after drop shape: ', allData2.shape)
    return allData2

# allData.count().describe([0.01,0.05,0.1,0.2,0.3,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.5,0.6,0.7,0.8]) #col count discribe
# allData.count(axis = 1).describe([0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]) # row count discribe

def find01Data(allData,labels): # return [dF with label = 1, dF with label = 0]
    # allData = dropnaByPrecent(allData,axisVal=1,nonNanPercentile=0.01)
    allData = allData.dropna(axis = 1, thresh = 17)
    userId1 = labels[labels == 1].index
    userId0 = labels[labels == 0].index
    return [allData.loc[userId1], allData.loc[userId0]]

# row count distributation, take 30 as treshold to drop and try
# count    55596.000000
# mean       126.773977
# std        105.446624
# min          6.000000
# 1%          16.000000
# 5%          18.000000
# 10%         18.000000
# 20%         30.000000
# 30%         42.000000
# 50%         90.000000
# max        769.000000
def modifyAllData(allData,labels):
    all1Data,all0Data = find01Data(allData,labels)
    all0Data = all0Data.dropna(axis=0, thresh = 90)
    # all0Data = dropnaByPrecent(all0Data,nonNanPercentile=0.01) # as 1 labels are too small, only drop 0 labeled data
    return[all1Data,all0Data]

def splitData(allData,labels):
    all1Data,all0Data = modifyAllData(allData,labels)
    all1DataTrain,all1DataVali = train_test_split(all1Data,test_size = 0.2)
    all0DataTrain,all0DataVali = train_test_split(all0Data,test_size = 0.2)
    allDataTrain = pd.concat([all1DataTrain,all0DataTrain])
    allDataVali = pd.concat([all1DataVali,all0DataVali])
    trainId = list(allDataTrain.index)
    valiId = list(allDataVali.index)
    trainLabel = labels.loc[trainId]
    valiLabel = labels.loc[valiId]
    return[allDataTrain,allDataVali,trainLabel,valiLabel]

def allDataOverSampling(allData,labels):
    ros = RandomOverSampler()
    allData,labels = ros.fit_sample(allData,labels)
    return [allData,labels]

def calAccu(result,label): # 1st parameter is result, 2nd is label
    label = [i[0] for i in label.values]
    p11 = sum(np.logical_and(result,label))
    p01 = sum(np.logical_and(label,np.logical_xor(label,result)))
    p00 = sum(np.logical_and(np.logical_not(result),np.logical_not(label)))
    p10 = sum(np.logical_and(np.logical_not(label),np.logical_xor(label,result)))
    return[p11,p01,p00,p10]

def calAccuTrain(result,label): # 1st parameter is result, 2nd is label
    label = [i for i in label]
    p11 = sum(np.logical_and(result,label))
    p01 = sum(np.logical_and(label,np.logical_xor(label,result)))
    p00 = sum(np.logical_and(np.logical_not(result),np.logical_not(label)))
    p10 = sum(np.logical_and(np.logical_not(label),np.logical_xor(label,result)))
    return[p11,p01,p00,p10]