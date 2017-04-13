import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from itertools import permutations, repeat, product

def readData(filename):
    filepath = './'+filename
    data = pd.read_csv(filepath,header=None)
    return data


def readUserInfo( ):
    user_info = readData("train/user_info_train.txt")
    user_info.columns = ['userId','gender','job','education','marriage','residentialType']
    user_info = user_info.set_index('userId').sort_index()
    return user_info

#读取用户银行账单表 对账单数据求和并返回
def readBankDetail( ):
    bank_detail = readData("train/bank_detail_train.txt")
    bank_detail.columns = ['userId','timeStamp','transType','transAmount','salaryIncome']
    bank_detail = bank_detail.set_index('userId').sort_index()
    return bank_detail

#读取信用卡账单记录 取均值并返回
def readBillDetail( ):
    bill_detail = readData("train/bill_detail_train.txt")
    bill_detail.columns = ['userId','timeStamp','bankId','lastBillAmt','lastPaidAmt','creditAmount','remainedBalThisMon',\
    'minPayThisMon','#ofTrans','balThisMon','ajtedAmt','evolInst','remainBal','cashCredictLimit','payStus']
    bill_detail = bill_detail.set_index('userId').sort_index()
    return bill_detail

#读取用户的浏览历史 对浏览数据求和并返回
def readBrowseHistory( ):
    browse_history = readData("train/browse_history_train.txt")
    browse_history.columns = ['userId','timeStamp','browsAct','browsId']
    browse_history = browse_history.set_index('userId').sort_index()
    return browse_history

#读取用户发放贷款时间 并返回
def readLoanTime( ):
    loan_time = readData("train/loan_time_train.txt")
    loan_time.columns = ['userId','timeStamp']
    loan_time = loan_time.set_index('userId').sort_index()
    return loan_time

 #读取类别信息
def readTarget( ):
    target = readData("train/overdue_train.txt")
    target.columns = ['userId', 'overDueLabel']
    target = target.set_index('userId').sort_index()
    return target

# def allData():
# allData = [readUserInfo(), readBankDetail(),readBillDetail(),readBrowseHistory(),readLoanTime(),readTarget()]
# allData = [readUserInfo(), readBankDetail(), readTarget()]
userInfo = readUserInfo()
bankDetail = readBankDetail()
billDetail = readBillDetail()
browseHist = readBrowseHistory()
loanTime = readLoanTime()
target = readTarget()
# all data above have timeStamp in Sec

# ==== change 2 layer indexing to 1 layer indexing ====

def flat2LevelIndex(df):
    colNames = list(product(df.index.levels[1],df.columns))
    returnDf = pd.DataFrame(np.ones((df.index.levels[0].shape[0],len(colNames))),index = df.index.levels[0],columns = colNames)
    returnDf[returnDf == 1] = np.nan
    levelOneIdx = df.index.levels[0]
    returnDf = pd.DataFrame(np.ones((df.index.levels[0].shape[0],len(colNames))),index = df.index.levels[0],columns = colNames)
    flatten = lambda x : pd.Series(x.values.flatten(),index = list(product(x.index,x.columns)))
    for idxL1 in list(levelOneIdx):
        returnDf.loc[idxL1] = flatten(df.loc[idxL1])
    return(returnDf)

# get bankDetail information
def bankDetailsEng(bankDetail):
    idx = pd.IndexSlice # claim idx for multiIndexing
    # bd = bankDetail.loc[[1,3]].copy()
    bd = bankDetail.copy()
    bd['timeStamp'] = bd['timeStamp']//2638000 # change to monthly time stamp
    bd = bd.reset_index()
    # bd1 = bd.groupby(['userId','timeStamp','transType','salaryIncome'])
    bd1 = bd.groupby(['userId','timeStamp','transType']).sum()
    income = bd1.loc[idx[:,:,0],idx['transAmount']] # transType == 0 for income
    spend = bd1.loc[idx[:,:,1],idx['transAmount']] # transType == 1 for spend

    bd1 = bd.groupby(['userId','timeStamp','salaryIncome']).sum()
    salary = bd1.loc[idx[:,:,1],idx[:]] # salaryIncome == 1 for salary
    income = income.reset_index().loc[:,['userId','timeStamp','transAmount']].set_index(['userId','timeStamp'])
    income.columns = ['totalIncome']
    spend = spend.reset_index().loc[:,['userId','timeStamp','transAmount']].set_index(['userId','timeStamp'])
    spend.columns = ['spend']
    salary = salary.reset_index().loc[:,['userId','timeStamp','transAmount']].set_index(['userId','timeStamp'])
    salary.columns = ['salary']
    salary.fillna(0)
    # income.to_csv('../newFeatures/income.csv')
    # spend.to_csv('../newFeatures/spend.csv')
    # salary.to_csv('../newFeatures/salary.csv')

    bankTotal = pd.concat([income,spend,salary],axis = 1)
    bankTotal['salary'] = bankTotal['salary'].fillna(0)

# get information of billDetail in monthley manner
def billDetailEng(billDetail):
    idx = pd.IndexSlice # claim idx for multiIndexing
    # bd = billDetail.loc[[2,4]].copy()
    bd = billDetail.copy()
    bd['timeStamp'] = bd['timeStamp']//2638000 # change to monthly time stamp
    bd = bd.reset_index()
    # bd1 = bd.groupby(['userId','timeStamp','transType','salaryIncome'])
    bd1 = bd.groupby(['userId','timeStamp']).sum()
    lastBillAmt = bd1['lastBillAmt']
    lastPaidAmt = bd1['lastPaidAmt']
    creditAmount = bd1['creditAmount']
    remainedBalThisMon = bd1['remainedBalThisMon']
    minPayThisMon = bd1['minPayThisMon']
    numOfTrans = bd1['#ofTrans']
    balThisMon = bd1['balThisMon']
    ajtedAmt = bd1['ajtedAmt']
    evolInst = bd1['evolInst']
    remainBal = bd1['remainBal']
    cashCredictLimit = bd1['cashCredictLimit']
    payStus = bd1['payStus']
    billTotal = pd.concat([lastBillAmt,lastPaidAmt,creditAmount,remainedBalThisMon,minPayThisMon,numOfTrans,balThisMon,ajtedAmt,evolInst,remainBal,cashCredictLimit,payStus],axis = 1)
    billTotal.to_csv('../newFeatures/billTotal.csv')
    return billTotal

# get info browseHist
# len(browseHist.browsAct.unique()) = 214
# len(browseHist.browsId.unique()) = 11
def browseHistEng(browseHist):
    bd = browseHist.loc[[2,4]].copy()
    bd = browseHist.copy()
    bd['timeStamp'] = bd['timeStamp']//2628000
    bd1 = bd.reset_index().groupby(['userId','timeStamp','browsId']).describe()
    # bd1.loc[idx[:,:,1],]

    browsIdCount = bd.reset_index().groupby(['userId','browsId']).count()['timeStamp'].reset_index()
    userIds = browsIdCount['userId']
    browsIds = browsIdCount['browsId']
    browsIdCount = browsIdCount.set_index(['userId','browsId'])
    browsIdCount.columns = ['browsIdCount']
    return

# i =0
# idCount = pd.DataFrame(columns=['userId'])
# for uId in userIds:
#     for browsId in browsIds:


moneyTotal = pd.concat([bankTotal,billTotal],axis = 1)
moneyTotalFlat = flat2LevelIndex(moneyTotal)
moneyTotal.to_csv('../newFeatures/moneyTotal1.csv')
moneyTotalFlat1 = flat2LevelIndex(moneyTotal)
moneyTotalFlat1 = moneyTotalFlat1.dropna(axis = 1,how='all')
moneyTotalFlat1.to_csv('../newFeatures/moneyTotalFlat1.csv')
moneyTotalFlat2 = pd.concat([bankTotalFlat,billTotalFlat], axis = 1)
moneyTotalFlat2.to_csv('../newFeatures/moneyTotalFlat2.csv')

#drop features if the whole row/column is NA
moneyTotalFlat1 = pd.read_csv('../newFeatures/moneyTotalFlat1.csv')
moneyTotalFlat1 = moneyTotalFlat1.set_index('userId')
moneyTotalFlat1 = moneyTotalFlat1.dropna(axis = 1, how = 'all')
moneyTotalFlat1 = moneyTotalFlat1.dropna(axis = 0, how = 'all')
moneyTotalFlat2 = pd.read_csv('../newFeatures/moneyTotalFlat2.csv')
moneyTotalFlat2 = moneyTotalFlat2.set_index('userId')
moneyTotalFlat2 = moneyTotalFlat2.dropna(axis = 1, how = 'all')
moneyTotalFlat2 = moneyTotalFlat2.dropna(axis = 0, how = 'all')
'''found in the end, there are total 73 months'''

tBank = bankDetail['timeStamp']
tBill = billDetail['timeStamp']
tBrowse = browseHist['timeStamp']
tLoan = loanTime['timeStamp']
totalT = pd.concat([tBank,tBill,tBrow)se,tLoan],ignore_index = True)
tStep = totalT.unique();
print(tStep.shape)
tDayStep = pd.Series(tStep//86400)
tDayStep = tDayStep.unique()
print(tDayStep.shape)
tMonStep = pd.Series(tDayStep //(365/12))
tMonStep = tMonStep.unique()
print(tMonStep.shape)

# allData = pd.concat(allData, axis = 1)
# allData.to_csv('allData.csv')