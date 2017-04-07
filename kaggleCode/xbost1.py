import xgboost as xgb
import numpy as np
import pandas as pd
### data preparation
from sklearn.cross_validation import train_test_split
import string
targetNums = list(range(1,27)) * 1000
letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))
fullData = pd.read_csv('train.csv')
dataId = fullData['Id']
target = fullData['Prediction']
target = target.map(letter2NumMap)
data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)
trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 20
clf = xgb.XGBClassifier(max_depth=30, learning_rate=0.001,
    n_estimators=300, silent=True,
    objective='logistic',
    nthread=-1, gamma=0, min_child_weight=1,
    max_delta_step=0, subsample=1, colsample_bytree=1,
    colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
    scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
clf.fit(trainData,trainTarget)
#bst = xgb.train(np.array(trainData), np.array(trainTarget), num_round)
# make prediction
preds = clf.predict(testData)
accuracies = sum(np.array(preds) == np.array(testTarget))/len(preds)
a=pd.Series(accuracies)
a.to_csv('oneResult.csv')
