# After loan time
bankAfterLoan = bankDetail2[bankDetail2['timeStamp']>bankDetail2['loanTime']] # get the transactions After loanTime

# count of incomes, amout of income, After loan
incomeGpAfter = bankAfterLoan[bankAfterLoan['transType'] == 0].groupby('userId',as_index = False)
incomeAfterLoan = incomeGpAfter['transAmount'].agg({'incomeCountAfterLoan':'count','totalIncomeAfterLoan':'sum'})
incomeAfterLoan = pd.merge(incomeAfterLoan, incomeGpAfter['timeStamp'].agg({'firstIncomeDay':'min','lastIncomeDay':'max'}), how='left', on = 'userId')
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
    /((featureAfterLoan['lastIncomeDay'] - featureAfterLoan['firstIncomeDay'])/365 * 12)
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['avgIncomeAfterLoan'] )
# plt.show()

# average Income per month
featureAfterLoan['avgSalaryIncomeAfterLoan'] = featureAfterLoan['totalSalaryAfterLoan']\
    /((featureAfterLoan['lastIncomeDay'] - featureAfterLoan['firstIncomeDay'])/365 * 12)
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['avgIncomeAfterLoan'] )
# plt.plot(featureAfterLoan['userId'] ,featureAfterLoan['avgSalaryIncomeAfterLoan'] )
# plt.legend(loc='upper left')
# plt.show()

# salaryIncome percentile in income
featureAfterLoan['salaryIncomePercentileAfterLoan'] = featureAfterLoan['totalSalaryAfterLoan'] / featureAfterLoan['totalIncomeAfterLoan'].fillna(0)
