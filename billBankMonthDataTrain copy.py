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
import matplotlib.pyplot as plt
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

def split3Data(allData,labels):
    all1Data,all0Data = modifyAllData(allData,labels)
    all1DataTrain,all1DataTest = train_test_split(all1Data,test_size = 0.2)
    all0DataTrain,all0DataTest = train_test_split(all0Data,test_size = 0.2)
    all1DataTrain,all1DataVali = train_test_split(all1DataTrain,test_size = 0.25)
    all0DataTrain,all0DataVali = train_test_split(all0DataTrain,test_size = 0.25)
    allDataTrain = pd.concat([all1DataTrain,all0DataTrain])
    allDataVali = pd.concat([all1DataVali,all0DataVali])
    allDataTest = pd.concat([all1DataTest,all0DataTest])
    trainId = list(allDataTrain.index)
    valiId = list(allDataVali.index)
    testId = list(allDataTest.index)
    trainLabel = labels.loc[trainId]
    valiLabel = labels.loc[valiId]
    testLabel = labels.loc[testId]
    return[allDataTrain,allDataVali,allDataTest,trainLabel,valiLabel,testLabel]

def allDataOverSampling(allData,labels):
    ros = RandomOverSampler()
    # allData,labels = ros.fit_sample(allData,labels,random_state = 1000)
    allData,labels = ros.fit_sample(allData,labels)
    return [allData,labels]


def find01Data(allData,labels): # return [dF with label = 1, dF with label = 0]
    # allData = dropnaByPrecent(allData,axisVal=1,nonNanPercentile=0.01)
    allData = allData.dropna(axis = 1, thresh = 5263)
    userId1 = labels[labels == 1].index
    userId0 = labels[labels == 0].index
    return [allData.loc[userId1], allData.loc[userId0]]


def modifyAllData(allData,labels):
    all1Data,all0Data = find01Data(allData,labels)
    all0Data = all0Data.dropna(axis=0, thresh = 132)
    # all0Data = dropnaByPrecent(all0Data,nonNanPercentile=0.01) # as 1 labels are too small, only drop 0 labeled data
    return[all1Data,all0Data]

# def testAlgo(allData, labels):
#================================================================================================================
#================================================================================================================
#================================================================================================================
allDataTrain, allDataVali, trainLabel, valiLabel = splitData(allData,labels)
allDataTrain,allDataVali,allDataTest,trainLabel,valiLabel,testLabel = split3Data(allData,labels)
allDataTrain = allDataTrain.fillna(allDataTrain.mean())
allDataTrain = allDataTrain.fillna(-1)
allDataVali = allDataVali.fillna(allDataVali.mean())
allDataVali = allDataVali.fillna(-1)
allDataTest = allDataTest.fillna(allDataVali.mean())
allDataTest = allDataTest.fillna(-1)
print(allDataTrain.shape)
allDataTrain,trainLabel = allDataOverSampling(allDataTrain,trainLabel)
print(allDataTrain.shape)
rf = RandomForestClassifier(n_estimators=50, min_impurity_split = 1e-8)
rf.fit(allDataTrain,trainLabel)
valiResult = rf.predict(allDataVali)
trainResult = rf.predict(allDataTrain)
testResult = rf.predict(allDataTest)
print(ks(valiResult,valiLabel))
print(ks(trainResult,trainLabel))
print(ks(testResult,testLabel))



for i in range(10):
    rf.fit(allDataTrain, trainLabel)
    valiResult = rf.predict(allDataVali)
    trainResult = rf.predict(allDataTrain)
    print(ks(valiResult, valiLabel))
    print(ks(trainResult, trainLabel))
    joblib.dump(rf, '../results/loopTest'+str(i)+'.pkl')

rfAll = rf
#================================================================================================================
#=========================================take out useful columns================================================
#================================================================================================================
featureCol = allDataVali.columns
rfImp = rfAll.feature_importances_
useAbleCol = featureCol[rfImp> pd.DataFrame(rfImp).describe([0.3]).loc['30%'].loc[0]]
allDataTrain, allDataVali, trainLabel, valiLabel = splitData(allData[useAbleCol],labels)
allDataTrain,allDataVali,allDataTest,trainLabel,valiLabel,testLabel = split3Data(allData[useAbleCol],labels)
#================================================================================================================
#================================================================================================================
#================================================================================================================
# return [allDataVali, valiResult,valiLabel,trainResult,trainLabel,rf]

# np.savetxt("../results/trainData.csv", allDataTrain, delimiter=",")
# np.savetxt("../results/trainLabel.csv", trainLabel, delimiter=",")
colValidProtion = allData.count().describe(np.arange(0,1,0.01)) #col count discribe
rowValidProtion = allData.count(axis = 1).describe(np.arange(0,1,0.01)) # row count discribe

def storeResults(rf,trainData,trainLabel,valiData,valiLabel):
    joblib.dump(rf,'../results/rf1.pkl')
    # rf2 = joblib.load('../results/rf1.pkl')
    # trainData.to_csv('../results/trainData.csv')
    # trainLabel.to_csv('../results/trainLabel.csv')
    np.savetxt("../results/trainData.csv", trainData, delimiter=",")
    np.savetxt("../results/trainLabel.csv", trainLabel, delimiter=",")
    valiData.to_csv('../results/valiData.csv')
    valiLabel.to_csv('../results/valiLabel.csv')
    return
rf = joblib.load('../results/rf1.pkl')
storeResults(rf,allDataTrain,trainLabel,allDataVali,valiLabel)

allDataTrain = np.genfromtxt('../results/trainData.csv',delimiter=',')
trainLabel = np.genfromtxt('../results/trainLabel.csv',delimiter=',')
allDataVali = pd.read_csv('../results/valiData.csv').set_index('userId')
valiLabel = pd.read_csv('../results/valiLabel.csv').set_index('userId')