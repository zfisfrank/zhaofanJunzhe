#/usr/bin/python3
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import string
from joblib import Parallel, delayed
#
# targetNums = list(range(1,27)) * 1000
# letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
# num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))

fullData = pd.read_csv('userInfoTrain.csv')

dataId = fullData['userId']
target = fullData['overDueLabel']
data = fullData.drop(['userId', 'overDueLabel'],axis = 1)
#data = fullData.drop(['Id', 'Prediction'],axis = 1)

"""this part to test current algorithms' accuracies"""
accuracies = []


trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)

# clf = svm.SVC(C=1,gamma=0.01)
def svm_fit(i,trainData, testData, trainTarget, testTarget):
    print('start loop no. : ' + str(i))
    gammaList = pd.read_csv('paraList.csv')
    #clf = svm.SVC(gamma = gammaList.iloc[i,1],C = gammaList.iloc[i,0],tol = 1e-5,coef0 = 0.1,kernel = 'sigmoid')
    #old one still work
    clf = svm.SVC(gamma = gammaList.iloc[i,1],C = gammaList.iloc[i,0],tol = 1e-2)
    clf.fit(trainData,trainTarget)
    Predictions = clf.predict(testData)
    accuracy = sum(Predictions == testTarget)/len(testTarget)
    outString = str(gammaList.iloc[i,1]) + ',' + str(gammaList.iloc[i,0]) + ',' +str(accuracy) + '\n'
    f= open("smvParaRunResult.txt","a+")
    f.write(outString)
    f.close()
    print(outString)
#    return accuracy
#
# gammaList = pd.read_csv('paraList.csv')
# Parallel(n_jobs=12)(delayed(svm_fit)(i,trainData, testData, trainTarget, testTarget) for i in range(len(gammaList)))
# for i in range(len(gammaList)):
#     svm_fit(i,trainData, testData, trainTarget, testTarget)

clf = RandomForestClassifier(n_estimators=20)
clf = svm.SVC()
clf.fit(trainData,trainTarget)
Predictions = clf.predict(testData)
accuracy = sum(Predictions == testTarget)/len(testTarget)
