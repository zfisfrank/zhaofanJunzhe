#/usr/bin/python3
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
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
def mlp_fit(i,trainData, testData, trainTarget, testTarget):
    print('start loop no. : ' + str(i))
    varList = pd.read_csv('MLPparaList.csv')
    # clf = svm.SVC(gamma = varList.iloc[i,1],C = varList.iloc[i,0])
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(400), random_state=10, beta_1 =varList.iloc[i,0] ,beta_2 =varList.iloc[i,1])
    clf.fit(trainData,trainTarget)
    Predictions = clf.predict(testData)
    accuracy = sum(Predictions == testTarget)/len(testTarget)
    outString = str(varList.iloc[i,0]) + ',' + str(varList.iloc[i,1]) + ',' +str(accuracy) + '\n'
    f= open("mlpParaRunResult.txt","a+")
    f.write(outString)
    f.close()
    print(outString)
#    return accuracy
# varList[0] is beta_1, varList[1] is beta_2
varList = pd.read_csv('MLPparaList.csv')
# for i in range(len(varList)):
#     mlp_fit(i,trainData, testData, trainTarget, testTarget)
Parallel(n_jobs=8)(delayed(mlp_fit)(i,trainData, testData, trainTarget, testTarget) for i in range(len(varList)))
# for i in range(len(varList)):
#     svm_fit(i,trainData, testData, trainTarget, testTarget)

# acc = pd.Series(accuracies)
# acc.to_csv('results2.txt')
#print(layers)
# print(accuracies)
