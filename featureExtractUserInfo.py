#usr/bin/python3
# 提取了 user_info_train, loan_time_train,overdue_train  放在同一个csv file 里面
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
# import string
from joblib import Parallel, delayed
from sklearn import preprocessing

# load the different values from csv

# 用户基本属性： 用户id,性别,职业,教育程度,婚姻状态,户口类型
userInfoCol = ['userId','gender','job','education','marriage','residentialType']
userInfo = pd.read_csv('../initData/user_info_train.txt',index_col = False, header  = None, names = userInfoCol)

# 放款时间信息: 用户id,放款时间
loanTimeCol = ['userId','loanTime']
loanTime = pd.read_csv('../initData/loan_time_train.txt',index_col = False, header  = None, names = loanTimeCol)
loanTime['loanTime'] = loanTime['loanTime'] // 86400
# user one loan time match with one userId, merge two together
userInfo = pd.merge(userInfo, loanTime, how='inner',on = "userId")

# 逾期行为的记录: 用户id,样本标签
overDueCol = ['userId','overDueLabel']
overDue = pd.read_csv('../initData/overdue_train.txt',index_col = False, header  = None,names = overDueCol)

# user one loan time match with one userId, merge two together
userInfo = pd.merge(userInfo, overDue, how='inner',on = "userId")

# userInfo.columns : 'userId', 'gender', 'job', 'education', 'marriage', 'residentialType','loanTime', 'overDueLabel'
userInfo = userInfo.set_index('userId').sort_index()
userInfo.to_csv('../dataSets/userInfoTrain2.csv')

# userInfo.columns : 'userId', 'gender', 'job', 'education', 'marriage', 'residentialType','loanTime', 'overDueLabel'
''' now check the precentile of each type of feature's precentile of overdue '''
testCols = ['gender', 'job', 'education', 'marriage', 'residentialType']
t = []
for colName in testCols:
    val = userInfo.groupby(colName)['overDueLabel'].sum()/userInfo.groupby(colName)['overDueLabel'].count()
    val.name = colName
    t.append(val)
# userInfo.groupby('gender')['overDueLabel'].sum()/userInfo.groupby('gender')['overDueLabel'].count()
t = pd.DataFrame(t).T
t.to_csv('featuresOverduePrecentage1.csv')
