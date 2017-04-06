#usr/bin/python3
# 提取了 bank_detail_train, put information into one row per userId
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ==========================================================================================================================#
# ==========================================================================================================================#
# ============================================features before the loan time=================================================#
# ==========================================================================================================================#
# ==========================================================================================================================#
# before loan time
bankBeforeLoan = bankDetail2[bankDetail2['timeStamp']<=bankDetail2['loanTime']] # get the transactions before loanTime

# count of incomes, amout of income, before loan
incomeGpBefore = bankBeforeLoan[bankBeforeLoan['transType'] == 0].groupby('userId',as_index = False)
incomeBeforeLoan = incomeGpBefore['transAmount'].agg({'incomeCountBeforeLoan':'count','totalIncomeBeforeLoan':'sum'})
incomeBeforeLoan = pd.merge(incomeBeforeLoan, incomeGpBefore['timeStamp'].agg({'firstIncomeDayBeforeLoan':'min','lastIncomeDay':'max'}), how='left', on = 'userId')
# count of spend, amount of spend, before loan
spendGpBefore = bankBeforeLoan[bankBeforeLoan['transType'] == 1].groupby('userId',as_index = False)
spendBeforeLoan = spendGpBefore['transAmount'].agg({'spendCountBeforeLoan':'count','totalSpendBeforeLoan':'sum'})

# count of salaryIncome,amount of salaryIncome, before loan
salaryGpBefore = bankBeforeLoan[bankBeforeLoan['salaryIncome'] == 1].groupby('userId',as_index = False)
salaryBeforeLoan = salaryGpBefore['transAmount'].agg({'salaryCountBeforeLoan':'count','totalSalaryBeforeLoan':'sum'})

# the features BeforeLoan
featureBeforeLoan = pd.merge(incomeBeforeLoan,spendBeforeLoan,how = 'outer', on = 'userId')
featureBeforeLoan = pd.merge(featureBeforeLoan,salaryBeforeLoan,how = 'outer', on = 'userId')
featureBeforeLoan = featureBeforeLoan.fillna(0) # if there is NaN, meanig the amount is 0



''' calculate some features according to uppper features '''

# calculate non salary income
featureBeforeLoan['nonSalaryIncomeBeforeLoan'] = featureBeforeLoan['totalIncomeBeforeLoan'] - featureBeforeLoan['totalSalaryBeforeLoan']
# plt.plot(featureBeforeLoan['userId'] ,featureBeforeLoan['nonSalaryIncomeBeforeLoan'] )
# plt.show()
# Overspend
featureBeforeLoan['overSpendBeforeLoan'] = featureBeforeLoan['totalSpendBeforeLoan'] - featureBeforeLoan['totalIncomeBeforeLoan']
# plt.plot(featureBeforeLoan['userId'] ,featureBeforeLoan['overSpendBeforeLoan'] )
# plt.show()

# average Income per month
featureBeforeLoan['avgIncomeBeforeLoan'] = featureBeforeLoan['totalIncomeBeforeLoan']\
    /((featureBeforeLoan['lastIncomeDay'] - featureBeforeLoan['firstIncomeDayBeforeLoan'])/365 * 12)
# plt.plot(featureBeforeLoan['userId'] ,featureBeforeLoan['avgIncomeBeforeLoan'] )
# plt.show()

# average salary per month
featureBeforeLoan['avgSalaryIncomeBeforeLoan'] = featureBeforeLoan['totalSalaryBeforeLoan']\
    /((featureBeforeLoan['lastIncomeDay'] - featureBeforeLoan['firstIncomeDayBeforeLoan'])/365 * 12)
# plt.plot(featureBeforeLoan['userId'] ,featureBeforeLoan['avgIncomeBeforeLoan'] )
# plt.plot(featureBeforeLoan['userId'] ,featureBeforeLoan['avgSalaryIncomeBeforeLoan'] )
# plt.legend(loc='upper left')
# plt.show()

# salaryIncome percentile in income
featureBeforeLoan['salaryIncomePercentileBeforeLoan'] = featureBeforeLoan['totalSalaryBeforeLoan'] / featureBeforeLoan['totalIncomeBeforeLoan'].fillna(0)


# ==========================================================================================================================#
# ==========================================================================================================================#
# ============================================features after the loan time=================================================#
# ==========================================================================================================================#
# ==========================================================================================================================#
# After loan time
bankAfterLoan = bankDetail2[bankDetail2['timeStamp']>bankDetail2['loanTime']] # get the transactions After loanTime

# count of incomes, amout of income, After loan
incomeGpAfter = bankAfterLoan[bankAfterLoan['transType'] == 0].groupby('userId',as_index = False)
incomeAfterLoan = incomeGpAfter['transAmount'].agg({'incomeCountAfterLoan':'count','totalIncomeAfterLoan':'sum'})
incomeAfterLoan = pd.merge(incomeAfterLoan, incomeGpAfter['timeStamp'].agg({'firstIncomeDayAfterLoan':'min','lastIncomeDay':'max'}), how='left', on = 'userId')
# count of spend, amount of spend, After loan
spendGpAfter = bankAfterLoan[bankAfterLoan['transType'] == 1].groupby('userId',as_index = False)
spendAfterLoan = spendGpAfter['transAmount'].agg({'spendCountAfterLoan':'count','totalSpendAfterLoan':'sum'})

# count of salaryIncome,amount of salaryIncome, After loan
salaryGpAfter = bankAfterLoan[bankAfterLoan['salaryIncome'] == 1].groupby('userId',as_index = False)
salaryAfterLoan = salaryGpAfter['transAmount'].agg({'salaryCountAfterLoan':'count','totalSalaryAfterLoan':'sum'})

# the features AfterLoan
featureAfterLoan = pd.merge(incomeAfterLoan,spendAfterLoan,how = 'outer', on = 'userId')
featureAfterLoan = pd.merge(featureAfterLoan,salaryAfterLoan,how = 'outer', on = 'userId')
featureAfterLoan = featureAfterLoan.fillna(0) # if there is NaN, meanig the amount is 0

# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['totalIncomeAfterLoan'] )
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['totalSalaryAfterLoan'] )
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['salaryIncomePercentileAfterLoan'] )
# plt.show()

''' calculate some features according to uppper features '''

# calculate non salary income
featureAfterLoan['nonSalaryIncomeAfterLoan'] = featureAfterLoan['totalIncomeAfterLoan'] - featureAfterLoan['totalSalaryAfterLoan']
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['nonSalaryIncomeAfterLoan'] )
# plt.show()
# Overspend
featureAfterLoan['overSpendAfterLoan'] = featureAfterLoan['totalSpendAfterLoan'] - featureAfterLoan['totalIncomeAfterLoan']
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['overSpendAfterLoan'] )
# plt.show()

# average Income per month
featureAfterLoan['avgIncomeAfterLoan'] = featureAfterLoan['totalIncomeAfterLoan']\
    /((featureAfterLoan['lastIncomeDay'] - featureAfterLoan['firstIncomeDayAfterLoan'])/365 * 12)
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['avgIncomeAfterLoan'] )
# plt.show()

# average Income per month
featureAfterLoan['avgSalaryIncomeAfterLoan'] = featureAfterLoan['totalSalaryAfterLoan']\
    /((featureAfterLoan['lastIncomeDay'] - featureAfterLoan['firstIncomeDayAfterLoan'])/365 * 12)
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['avgIncomeAfterLoan'] )
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['avgSalaryIncomeAfterLoan'] )
# plt.legend(loc='upper left')
# plt.show()

# salaryIncome percentile in income
featureAfterLoan['salaryIncomePercentileAfterLoan'] = featureAfterLoan['totalSalaryAfterLoan'] / featureAfterLoan['totalIncomeAfterLoan'].fillna(0)
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['salaryIncomePercentileAfterLoan'] )
# plt.show()

features = pd.merge(featureBeforeLoan,featureAfterLoan,how = 'outer', on ='userId')
