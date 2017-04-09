# /usr/local/bin/python3

import os
import numpy as np
import pandas as pd
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
# from pandas import DataFrame
# from sklearn.model_selection import ShuffleSplit
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVC
import random
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# import xgboost as xgb
# plot feature importance manually
# from numpy import loadtxt
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# from scipy.stats import ks_2samp

# ks value prediction method
#Kolmogorov-Smirnov statistics (KS)
from sklearn import metrics
def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()

# read the features
billDetailFeaturesTrain = pd.read_csv(
    '../featureFolderTrain/billDetailFeaturesTrain.csv',index_col = 0).set_index('userId')
bankDetailsFeaturesTrain = pd.read_csv(
    '../featureFolderTrain/bankDetailsFeaturesTrain.csv',index_col = 0).set_index('userId')

browseHistFeaturesTrain = pd.read_csv(
    '../featureFolderTrain/browseHistFeaturesTrain.csv',index_col = 0).set_index('userId')
fullInfoTrain = pd.read_csv(
    '../featureFolderTrain/fullInfoTrain.csv',index_col = 0).set_index('userId')
#
# pd.Series(fullInfoTrain.columns).align(pd.Series(bankDetailsFeaturesTrain.columns),join = 'inner')
fullInfoTrain = fullInfoTrain['overDueLabel']
fullInfoTrain = fullInfoTrain.dropna()
# bankDetailsFeaturesTrain.drop(list(fullInfoTrain.columns),axis = 1)

allData = pd.concat([billDetailFeaturesTrain, bankDetailsFeaturesTrain, browseHistFeaturesTrain],axis = 1, join = 'outer')
allData = allData.join(fullInfoTrain,how = 'right')
allData = allData.replace(np.inf, np.nan)
allData = allData.dropna(axis = 1, how = 'all')
allData = allData.fillna(allData.mean())

# allData.to_csv('../featureFolderTrain/allTrain.csv')
allData = pd.read_csv('../featureFolderTrain/allTrain.csv')
allData = allData.drop('loanTime',axis = 1)


def testAlgo(allData):
    trainData,valiData = train_test_split(allData,test_size = 0.2)
    trainTarget = trainData['overDueLabel']
    trainFeatures = trainData.drop('overDueLabel',axis = 1)
    valiTarget = valiData['overDueLabel']
    valiFeatures = valiData.drop('overDueLabel',axis = 1)
    rf = RandomForestClassifier(n_estimators=50)
    # rf.fit(trainFeatures.set_index('userId').sort_index(),trainTarget.set_index('userId').sort_index())
    rf.fit(trainFeatures,trainTarget)
    # print(rf.feature_importances_)
    # print(pd.DataFrame(rf.feature_importances_).describe([0.5,0.6,0.7,0.8,0.9,0.95,0.99]))
    # pd.DataFrame(rf.feature_importances_).describe([0.5])
    valiResult = rf.predict(valiFeatures)
    trainResult = rf.predict(trainFeatures)
    return[valiFeatures.columns, valiResult,valiTarget,trainResult,trainTarget,rf]

print(ks(valiResult,valiTarget))
print(ks(trainResult,trainTarget))

imptIdx = rfAll.feature_importances_
useAbleCol = featureCol[rfAll.feature_importances_>pd.DataFrame(rfAll.feature_importances_).describe([0.9]).loc['90%'].loc[0]]
useAbleCol = list(useAbleCol)
useAbleCol.append('overDueLabel')
adjuestData = allData[useAbleCol]
[featureCol,valiResult,valiTarget,trainResult,trainTarget,rf] = testAlgo(adjuestData)
# [valiResult,trainTarget,rf] = result
[featureCol,valiResult,valiTarget,trainResult,trainTarget,rfAll] = testAlgo(allData)
