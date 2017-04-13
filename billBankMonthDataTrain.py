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

def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()

    # read the datas
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
    # fullInfoTrain = pd.read_csv('../featureFolderTrain/fullInfoTrain.csv', index_col=0).set_index('userId')
    userInfo = pd.read_csv('train/user_info_train.txt', names=['userId','gender', 'job', 'education', 'marriage', 'residentialType']).set_index('userId')
    labels = pd.read_csv('train/overdue_train.txt',names =['userId','overDueLabel']).set_index('userId')
    loanTime = pd.read_csv('train/loan_time_train.txt',names =['userId','loanTime']).set_index('userId')
    fullInfo = pd.concat([userInfo,loanTime],axis = 1)
    # fullInfo = fullInfoTrain.drop('overDueLabel', axis = 1)
    # labels = fullInfoTrain['overDueLabel']
    return [fullInfo,labels]

def combineAllInfo():
    [moneyTotalFlat1, moneyTotalFlat2] = readMoneyData()
    [fullInfo,labels] = readUserInfo()
    allData = pd.concat([fullInfo,moneyTotalFlat1],axis = 1)
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

def find01Data(allData,labels): # return [dF with label = 1, dF with label = 0]
    userId1 = labels[labels == 1].index
    userId0 = labels[labels == 0].index
    return [allData.loc[userId1], allData.loc[userId0]]

# usable drop rate @ 0.0025
def modifyAllData(allData,labels):
    all1Data,all0Data = find01Data(allData,labels)
    all0Data = dropnaByPrecent(all0Data,nonNanPercentile=0.0001) # as 1 labels are too small, only drop 0 labeled data
    return[all1Data,all0Data]

def splitData(allData,labels):
    all1Data,all0Data = modifyAllData(allData,labels)
    all1DataTrain,all1DataVali = train_test_split(all1Data,test_size = 0.5)
    all0DataTrain,all0DataVali = train_test_split(all0Data,test_size = 0.5)
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

def testAlgo(allData, labels):
allDataTrain, allDataVali, trainLabel, valiLabel = splitData(allData,labels)

allDataTrain = allDataTrain.fillna(allDataTrain.mean())
allDataTrain = allDataTrain.fillna(-1)
allDataVali = allDataVali.fillna(allDataVali.mean())
allDataVali = allDataVali.fillna(-1)

allDataTrain,trainLabel = allDataOverSampling(allDataTrain,trainLabel)

rf = RandomForestClassifier(n_estimators=50, min_impurity_split = 1e-6)
# rf = SVC()
rf.fit(allDataTrain,trainLabel)
valiResult = rf.predict(allDataVali)
trainResult = rf.predict(allDataTrain)
print(ks(valiResult,valiLabel))
print(ks(trainResult,trainLabel))
return [allDataVali, valiResult,valiLabel,trainResult,trainLabel,rf]

