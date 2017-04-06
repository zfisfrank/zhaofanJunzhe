#usr/bin/python3
# 提取了 bank_detail_train, put information into one row per userId
import numpy as np
import pandas as pd

# loan information load
# userInfo.columns : 'userId', 'gender', 'job', 'education', 'marriage', 'residentialType','loanTime', 'overDueLabel'
userInfo = pd.read_csv('userInfoTrain.csv')

# 银行流水记录：用户id,时间戳,交易类型,交易金额,工资收入标记
bankDetailCol = ['userId','timeStamp','transType','transAmount','salaryIncome']
bankDetail = pd.read_csv('bank_detail_train.csv',index_col = False, header  = None, names = bankDetailCol)
bankDetail['timeStamp'] = bankDetail['timeStamp'] // 86400 # sec convert to day

''' convert all rows belong to one userId into one row'''
# check timeStamp==0, percentile is small: 0.00638743684925
print(sum(bankDetail['timeStamp'] == 0)/len(bankDetail))

# add loan time into the bankDetail
bankDetail2 = pd.merge(bankDetail,userInfo[['userId','loanTime']],how = 'left',on = 'userId')

# before loan time
bankBeforeLoan = bankDetail2[bankDetail2['timeStamp']<=bankDetail2['loanTime']] # get the transactions before loanTime

# count of incomes, amout of income, before loan
incomeGpBefore = bankBeforeLoan[bankBeforeLoan['transType'] == 0].groupby('userId',as_index = False)
incomeBeforeLoan = incomeGpBefore['transAmount'].agg({'incomeCountBeforeLoan':'count','totalIncomeBeforeLoan':'sum'})

# count of spend, amount of spend, before loan
spendGpBefore = bankBeforeLoan[bankBeforeLoan['transType'] == 1].groupby('userId',as_index = False)
spendBeforeLoan = spendGpBefore['transAmount'].agg({'spendCountBeforeLoan':'count','totalSpendBeforeLoan':'sum'})

# count of salaryIncome,amount of salaryIncome, before loan
salaryGpBefore = bankBeforeLoan[bankBeforeLoan['salaryIncome'] == 1].groupby('userId',as_index = False)
salaryBeforeLoan = salaryGpBefore['transAmount'].agg({'salaryCountBeforeLoan':'count','totalSalaryBeforeLoan':'sum'})

# the features BeforeLoan
featureBeforeLoan = pd.merge(incomeBeforeLoan,spendBeforeLoan,how = 'outer', on = 'userId')
featureBeforeLoan = pd.merge(featureBeforeLoan,salaryBeforeLoan,how = 'outer', on = 'userId')
featureBeforeLoan = featureBeforeLoan.fillna(0) #