#/usr/bin/python3
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn import svm
import string
from joblib import Parallel, delayed

targetNums = list(range(1,27)) * 1000
letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))

fullData = pd.read_csv('train.csv')

dataId = fullData['Id']
target = fullData['Prediction']
target = target.map(letter2NumMap)
data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)
#data = fullData.drop(['Id', 'Prediction'],axis = 1)

"""this part to test current algorithms' accuracies"""
accuracies = []


trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)

# clf = svm.SVC(C=1,gamma=0.01)
def svm_fit(i,trainData, testData, trainTarget, testTarget):
    print('start loop no. : ' + str(i))
    #gammaList = pd.read_csv('tolList.csv')
    #clf = svm.SVC(gamma = gammaList.iloc[i,1],C = gammaList.iloc[i,0],tol = 1e-5,coef0 = 0.1,kernel = 'sigmoid')
    #old one still work
    clf = svm.SVC(gamma = 0.06,C = 5,tol = i)
    clf.fit(trainData,trainTarget)
    Predictions = clf.predict(testData)
    accuracy = sum(Predictions == testTarget)/len(testTarget)
    # outString = str(gammaList.iloc[i,1]) + ',' + str(gammaList.iloc[i,0]) + ',' +str(accuracy) + '\n'
    outString = str(i) + ',' +str(accuracy) + '\n'
    f= open("smvParaRunResult.txt","a+")
    f.write(outString)
    f.close()
    print(outString)
#    return accuracy
# tolList = [0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002]
tolList = [0.0009,0.0008,0.0007,0.0006,0.0005,0.0004,0.0003,0.0002,0.0001,0.00009,0.00008,0.00007,0.00006,0.00005,0.00004,0.00003,0.00002,0.00001]
# gammaList = pd.read_csv('tolList.csv')
Parallel(n_jobs=12)(delayed(svm_fit)(i,trainData, testData, trainTarget, testTarget) for i in tolList)
# for i in range(len(gammaList)):
#     svm_fit(i,trainData, testData, trainTarget, testTarget)

# acc = pd.Series(accuracies)
# acc.to_csv('results2.txt')
#print(layers)
# print(accuracies)
