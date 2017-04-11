# /usr/local/bin/python3

import os
import numpy as np
import pandas as pd
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import xgboost as xgb
# plot feature importance manually
from numpy import loadtxt
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()

# read the features
billDetailFeaturesTrain = pd.read_csv('../featureFolderTrain/billDetailFeaturesTrain.csv')
billDetailFeaturesTrain = billDetailFeaturesTrain.drop('Unnamed: 0',axis = 1).sort_values('userId')
billDetailFeaturesTrain = billDetailFeaturesTrain.fillna(billDetailFeaturesTrain.mean())
# read target
totalTarget = pd.read_csv('../featureFolderTrain/fullInfoTrain.csv',index_col = 0).sort_values('userId')
totalTarget = totalTarget[['userId','overDueLabel']]
totalTarget = pd.DataFrame(totalTarget)
totalData = totalTarget.merge(billDetailFeaturesTrain,how = 'inner', on = 'userId' )


trainData,testData = train_test_split(totalData.set_index('userId').sort_index(),test_size = 0.5)

#
# totalTarget = totalData[['userId','overDueLabel']]
# totalFeatures = totalData.drop('overDueLabel',axis = 1)

# trainTarget = trainData[['userId','overDueLabel']]
trainTarget = trainData['overDueLabel']
trainFeatures = trainData.drop('overDueLabel',axis = 1)

testTarget = testData['overDueLabel']
testFeatures = testData.drop('overDueLabel',axis = 1)

rf = RandomForestClassifier(n_estimators=50)
# rf.fit(trainFeatures.set_index('userId').sort_index(),trainTarget.set_index('userId').sort_index())
rf.fit(trainFeatures,trainTarget)

print(rf.feature_importances_)

print(pd.DataFrame(rf.feature_importances_).describe([0.5,0.6,0.7,0.8,0.9,0.95,0.99]))

valiResult = rf.predict(testFeatures)
print(ks_2samp(testTarget,valiResult))

trainResult = rf.predict(trainFeatures)
print(ks_2samp(trainTarget,trainResult))

#Kolmogorov-Smirnov statistics (KS)
from sklearn import metrics


print(ks(valiResult,testTarget))
# print(ks_2samp(valiResult,testTarget))
print(ks(trainResult,trainTarget))
# print(ks_2samp(trainResult,trainTarget))

# trueFeatureTrain = trainFeatures[trainTarget==1]
# trueTargetTrain = trainTarget[trainTarget==1]
# rf.fit()

trueFeatureVali = testFeatures[testTarget == 1]
trueResultTest = rf.predict(trueFeatureVali)
# check the column contain 'TimeStmp'
# featureCol = pd.DataFrame(totalFeatures.columns)
# featureFilter = featureCol.applymap(lambda x : 'TimeStmp' in x)
# featureFilter = pd.Series(featureFilter)

# totalFeatures = totalFeatures.fillna(totalFeatures.mean())

# estimatoerCounts = [i for i in range(50, 100, 100)]
# for estimatoerCount in estimatoerCounts:




# f= open("billDetailFeaturesTrain.csv","a+")
# f.write()
# f.write(str(list(rf.feature_importances_)))
# f.write('\n\r')
# f.close()

# f= open("billDetailFeaturesTrainFeatureCol.csv","a+")
# f= open("billDetailFeaturesTrainFeatureCol.csv","a+")
# f.write(str(featureCol))
# f.close()
