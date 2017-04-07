#usr/bin/python3
#

# 提取了 用户基本属性，放款时间信息，逾期行为的记录， 放在同一个csv file 里面
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
userInfo = pd.read_csv('user_info_train.csv',index_col = False, header  = None, names = userInfoCol)

# 放款时间信息: 用户id,放款时间
loanTimeCol = ['userId','loanTime']
loanTime = pd.read_csv('loan_time_train.csv',index_col = False, header  = None, names = loanTimeCol)
# loanTime['loanTime'] = loanTime['loanTime'] // 86400
# user one loan time match with one userId, merge two together
userInfo = pd.merge(userInfo, loanTime, how='outer',on = "userId").sort_values('userId')

# 逾期行为的记录: 用户id,样本标签
overDueCol = ['userId','overDueLabel']
overDue = pd.read_csv('overdue_train.csv',index_col = False, header  = None,names = overDueCol).sort_values('userId')

# split the data into train 80% and test 20%
trainData, testData, trainTarget, testTarget = train_test_split(userInfo,overDue,test_size= .2)

trainId = trainData['userId']
testId = testData['userId']

# save to csv
trainData.to_csv('../dataSets/userInfoTrain.csv')
testData.to_csv('../dataSets/userInfoTest.csv')
trainTarget.to_csv('../dataSets/trainTarget.csv')
testTarget.to_csv('../dataSets/testTarget.csv')

fullInfo = pd.merge(userInfo,overDue,how = 'outer',on = 'userId')
fullInfoTrain = fullInfo.loc[trainId]
fullInfoTest = fullInfo.loc[testId]
fullInfoTrain.to_csv('../dataSets/fullInfoTrain.csv')
fullInfoTest.to_csv('../dataSets/fullInfoTest.csv')
# read bankDetail and split according to trainId/testId
# ================================================================================#
# ================================================================================#
# 银行流水记录：用户id,时间戳,交易类型,交易金额,工资收入标记
bankDetailCol = ['userId','timeStamp','transType','transAmount','salaryIncome']
bankDetail = pd.read_csv('bank_detail_train.csv',index_col = False, header  = None, names = bankDetailCol).set_index('userId')
bankDetailTrain = bankDetail.loc[trainId]
bankDetailTest = bankDetail.loc[testId]
# save to csv
bankDetailTrain.to_csv('../dataSets/bankDetailTrain.csv')
bankDetailTest.to_csv('../dataSets/bankDetailTest.csv')


# read billDetail and split according to trainId/testId
# ================================================================================#
# ================================================================================#
# 信用卡账单记录：用户id，账单时间戳，银行id，上期账单金额， 上期还款金额，信用卡额度，本期账单余额，
# 本期账单最低还款，消费笔数,本期账单金额，调整金额，循环利息，可用余额，预借现金额度， 还款状态
# possible noise : bankId
# 加入 用户使用年数； grouping by time range rows of bankDetail, billDetail of same user ;
billDetailCol = \
    ['userId','billTimeStmp','bankId','lastBillAmt','lastPaidAmt','creditAmount','remainedBalThisMon',\
    'minPayThisMon','#ofTrans','balThisMon','ajtedAmt','evolInst','remainBal','cashCredictLimit','payStus']
billDetail = pd.read_csv('bill_detail_train.csv',index_col = False, header  = None, names = billDetailCol).set_index('userId')
billDetailTrain = billDetail.loc[trainId]
billDetailTest = billDetail.loc[testId]
# save to csv
billDetailTrain.to_csv('../dataSets/billDetailTrain.csv')
billDetailTest.to_csv('../dataSets/billDetailTest.csv')

# read browsing history and split according to trainId/testId
# ================================================================================#
# ================================================================================#
 # 用户浏览行为: 用户id,时间戳,浏览行为数据,浏览子行为编号
browseHistCol = ['userId','timeStmp','browsAct','browsId']
browseHist = pd.read_csv('browse_history_train.csv',index_col = False, header  = None, names = browseHistCol).set_index('userId')
browseHistTrain = browseHist.loc[trainId]
browseHistTest = browseHist.loc[testId]
browseHistTrain.to_csv('../dataSets/browseHistTrain.csv')
browseHistTest.to_csv('../dataSets/browseHistTest.csv')
