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
userInfo = pd.read_csv('../initData/user_info_train.txt',index_col = False, header  = None, names = userInfoCol)

# 放款时间信息: 用户id,放款时间
loanTimeCol = ['userId','loanTime']
loanTime = pd.read_csv('../initData/loan_time_train.txt',index_col = False, header  = None, names = loanTimeCol)
# loanTime['loanTime'] = loanTime['loanTime'] // 86400
# user one loan time match with one userId, merge two together
userInfo = pd.merge(userInfo, loanTime, how='outer',on = "userId").sort_values('userId')

# 逾期行为的记录: 用户id,样本标签
overDueCol = ['userId','overDueLabel']
overDue = pd.read_csv('../initData/overdue_train.txt',index_col = False, header  = None,names = overDueCol).sort_values('userId')

userInfo0 = userInfo[overDue['overDueLabel']==0]
userInfo1 = userInfo[overDue['overDueLabel']==1]
# split the data into train 80% and test 20%
# trainData, testData, trainTarget, testTarget = train_test_split(userInfo,overDue,test_size= .2)

trainData0, testData0 = train_test_split(userInfo0,test_size= .2)
trainData1, testData1 = train_test_split(userInfo1,test_size= .2)

trainData = pd.concat([trainData0,trainData1]).set_index('userId').sort_index().reset_index()
testData = pd.concat([testData0,testData1]).set_index('userId').sort_index().reset_index()

trainId = trainData['userId']
testId = testData['userId']
trainTarget = overDue.set_index('userId').loc[trainId].sort_index().reset_index()
testTarget = overDue.set_index('userId').loc[testId].sort_index().reset_index()

# save to csv
trainData.to_csv('../dataSets/userInfoTrain.csv')
testData.to_csv('../dataSets/userInfoTest.csv')
trainTarget.to_csv('../dataSets/trainTarget.csv')
testTarget.to_csv('../dataSets/testTarget.csv')

fullInfo = pd.merge(userInfo,overDue,how = 'outer',on = 'userId') # should not matter, but inner join should be more approprate
fullInfoTrain = fullInfo.set_index('userId').loc[trainId].sort_index().reset_index()
fullInfoTest =fullInfo.set_index('userId').loc[testId].sort_index().reset_index()
fullInfoTrain['loanTime'] = fullInfoTrain['loanTime'] // 86400
fullInfoTest['loanTime'] = fullInfoTest['loanTime'] // 86400
fullInfoTrain.to_csv('../featureFolderTrain/fullInfoTrain.csv')
fullInfoTest.to_csv('../featureFolderTest/fullInfoTest.csv')
# read bankDetail and split according to trainId/testId
# ================================================================================#
# ================================================================================#
# 银行流水记录：用户id,时间戳,交易类型,交易金额,工资收入标记
bankDetailCol = ['userId','timeStamp','transType','transAmount','salaryIncome']
bankDetail = pd.read_csv('../initData/bank_detail_train.txt',index_col = False, header  = None, names = bankDetailCol).set_index('userId')
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
billDetail = pd.read_csv('../initData/bill_detail_train.txt',index_col = False, header  = None, names = billDetailCol).set_index('userId')
billDetailTrain = billDetail.loc[trainId].dropna(how = 'all')
billDetailTest = billDetail.loc[testId].dropna(how = 'all')
# save to csv
billDetailTrain.to_csv('../dataSets/billDetailTrain.csv')
billDetailTest.to_csv('../dataSets/billDetailTest.csv')

# read browsing history and split according to trainId/testId
# ================================================================================#
# ================================================================================#
 # 用户浏览行为: 用户id,时间戳,浏览行为数据,浏览子行为编号
browseHistCol = ['userId','timeStmp','browsAct','browsId']
browseHist = pd.read_csv('../initData/browse_history_train.txt',index_col = False, header  = None, names = browseHistCol).set_index('userId')
browseHistTrain = browseHist.loc[trainId].dropna(how = 'all')
browseHistTest = browseHist.loc[testId].dropna(how = 'all')
browseHistTrain.to_csv('../dataSets/browseHistTrain.csv')
browseHistTest.to_csv('../dataSets/browseHistTest.csv')