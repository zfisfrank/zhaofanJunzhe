
# coding: utf-8

# In[8]:

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
from xgboost import XGBClassifier
from matplotlib import pyplot

#load data into df
path_trainData1 = os.path.abspath('/Users/apple/Dropbox/CS5339/featureFolderTrain/fullInfoTrain.csv')
path_trainData2 = os.path.abspath('/Users/apple/Dropbox/CS5339/featureFolderTrain/bankDetailsFeaturesTrain.csv')

str(path_trainData1)
str(path_trainData2)
fullInfoTrain = pd.read_csv(path_trainData1) #can put header=1
bankDetailsFeaturesTrain = pd.read_csv(path_trainData2)
print 'fullInfoTrain.shape '+repr(fullInfoTrain.shape)
print 'bankDetailsFeaturesTrain.shape '+repr(bankDetailsFeaturesTrain.shape)
#print fullInfoTrain.head(30)
#print list(fullInfoTrain.columns.values)
#get dummies
fullInfoTrainDumm=pd.get_dummies(fullInfoTrain, prefix=None, prefix_sep='_', dummy_na=False,
                     columns=['gender','job','education','marriage','residentialType'], sparse=False)
#print list(fullInfoTrainDumm.columns.values)
fullInfoTrainSelc = fullInfoTrainDumm.loc[:,['Unnamed: 0', 'userId', 'loanTime', 'overDueLabel', 'gender_0.0', 'gender_1.0', 'gender_2.0', 
 'job_0.0', 'job_1.0', 'job_2.0', 'job_3.0', 'job_4.0', 'education_0.0', 'education_1.0', 
 'education_2.0', 'education_3.0', 'education_4.0', 'marriage_0.0', 'marriage_1.0', 'marriage_2.0', 
 'marriage_3.0', 'marriage_4.0', 'marriage_5.0', 'residentialType_0.0', 'residentialType_1.0', 
 'residentialType_2.0', 'residentialType_3.0', 'residentialType_4.0']]


# In[3]:

# drop NA
#data=data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# fill NA with mean
data=data.fillna(data.mean())
print "overall dataFram, filled in NaN with mean: "+repr(data.shape)

# random split
train, CV = train_test_split(data, test_size = 0.1)
print train.shape
print CV.shape

#print train.head(10)

y_train=train.loc[:,'overDueLabel']
X_train=train.loc[:,['gender','job','education','marriage','residentialType']]
y_CV=CV.loc[:,'overDueLabel']
X_CV=CV.loc[:,['gender','job','education','marriage','residentialType']]

#get dummies for X_train and X_CV


print "train set X DF, w/ dummy :"+repr(X_train.shape)
print "train set y DF w/ dummy :"+repr(y_train.shape)
print "CV set X DF w/dummy: "+repr(X_CV.shape)
#print X_train.head(20)

#convert to matrix
X_matrix=X_train.as_matrix()
y_matrix=y_train.as_matrix()
print "train set X matrix :"+repr(X_matrix.shape)
print "train set y matrix :"+repr(y_matrix.shape)

rf = RandomForestClassifier(n_estimators=50) 


# reg_alpha, reg_lambda
rf.fit(X_matrix,y_matrix)
pred_train=rf.predict(X_matrix)

pred_CV=rf.predict(X_CV)
Score=np.mean(pred_train==y_matrix)*100
Score2=np.mean(pred_CV==y_CV)*100
print "RandomForest result: %f" %Score
print "RandomForest result: %f" %Score2
print "number of overdue predicted:"+repr(np.sum(pred_train)+np.sum(pred_CV))
print "number of overdue exists: "+repr(np.sum(y_train)+np.sum(y_CV))


#Kolmogorov-Smirnov statistics (KS)
from sklearn import metrics
def ks(y_predicted, y_true):
    
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()

print ks(pred_train,y_matrix)


print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), colNames), reverse=True)
print 'done'


# In[4]:

['Unnamed: 0', 'userId', 'loanTime', 'overDueLabel', 'gender_0.0', 'gender_1.0', 'gender_2.0', 
 'job_0.0', 'job_1.0', 'job_2.0', 'job_3.0', 'job_4.0', 'education_0.0', 'education_1.0', 
 'education_2.0', 'education_3.0', 'education_4.0', 'marriage_0.0', 'marriage_1.0', 'marriage_2.0', 
 'marriage_3.0', 'marriage_4.0', 'marriage_5.0', 'residentialType_0.0', 'residentialType_1.0', 
 'residentialType_2.0', 'residentialType_3.0', 'residentialType_4.0']


# In[ ]:



