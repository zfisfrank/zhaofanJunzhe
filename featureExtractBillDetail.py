#usr/bin/python3
# 提取了 bank_detail_train, put information into one row per userId
import numpy as np
import pandas as pd

def doubleIndex2single(doubleDf):
    doubleDf = doubleDf.sort_values('userId')
    index1 = doubleDf.iloc[:,0].unique() # 1st level index
    index2 = doubleDf.iloc[:,1].unique() # 2nd level index
    index1Len = len(index1)
    index2Len = len(index2)
    doubleDf = doubleDf.set_index(list(doubleDf.columns[0:2]))
    colName = doubleDf.columns.values # column names
    colNameLen = len(colName)
    # flatten all value into 1-d array
    valueArray = doubleDf.values.flatten('C')
    valueArray = valueArray.reshape(index1Len, index2Len*colNameLen)
    # get the column names for the new dataframe
    newColName = []
    for idx1 in index2:
        # print(idx1)
        for idx2 in colName:
            newColName.append(str(idx1) + '_' + str(idx2))
    returnDf = pd.DataFrame(valueArray,columns = newColName)
    returnDf['userId'] = index1
    returnDf = returnDf.set_index('userId').reset_index()
    return returnDf

userInfo = pd.read_csv('../dataSets/userInfoTrain.csv').sort_values('userId')
userInfo['loanTime'] = userInfo['loanTime'] // 86400
# 信用卡账单记录：用户id，账单时间戳，银行id，上期账单金额， 上期还款金额，信用卡额度，本期账单余额，
# 本期账单最低还款，消费笔数,本期账单金额，调整金额，循环利息，可用余额，预借现金额度， 还款状态

billDetail = pd.read_csv('../dataSets/billDetailTrain.csv').sort_values('userId')
billDetail['billTimeStmp'] = billDetail['billTimeStmp'] // 86400
billDetail2 = pd.merge(billDetail,userInfo[['userId','loanTime']],how = 'right',on = 'userId')
billDetail2['unPaidedBill'] = billDetail2['lastBillAmt'] - billDetail2['lastPaidAmt']
billDetail2['spentCreditThisMonth'] = billDetail2['creditAmount'] - billDetail2['remainedBalThisMon']
billDetail2['remainBalVscashCreditLimit'] = billDetail2['remainBal'] - billDetail2['cashCredictLimit']

# ==========================================================================================================================#
# ==========================================================================================================================#
# ============================================features before the loan time=================================================#
# ==========================================================================================================================#
# ==========================================================================================================================#
billDetailBeforeLoan = billDetail2[billDetail2['billTimeStmp'] <= billDetail2['loanTime']].drop('loanTime',axis =1)
billDetailGpBeforeLoan = billDetailBeforeLoan.groupby('userId')
billFeatureBeforeLoan = billDetailGpBeforeLoan['minPayThisMon'].agg({'minMonthPaySum':'sum'}).reset_index()
billFeatureBeforeLoan =pd.merge(billFeatureBeforeLoan, billDetailGpBeforeLoan['evolInst'].agg({'evolInstSum':'sum'}).reset_index(),how = 'outer', on = 'userId')
# billDescribeBeforeLoan = billDetailGpBeforeLoan.describe()
# billDescribeBeforeLoan.to_csv('../describes/billDescribeBeforeLoan.csv')
# billDescribeBeforeLoan = billDescribeBeforeLoan.reset_index()
billDescribeBeforeLoan = pd.read_csv('../dataSets/billDescribeBeforeLoan.csv')
# a = doubleIndex2single(billDescribeBeforeLoan.reset_index())
billDescribeBeforeLoan = doubleIndex2single(billDescribeBeforeLoan)
billFeatureBeforeLoan = pd.merge(billFeatureBeforeLoan, billDescribeBeforeLoan, how  = 'outer', on = 'userId')
colNamesBeforeLoan = ['userId']
colNamesBeforeLoan += (list(billFeatureBeforeLoan.columns[1:]+ 'BeforeLoan'))
billFeatureBeforeLoan.columns = colNamesBeforeLoan

billFeatureBeforeLoan = pd.read_csv('billFeatureBeforeLoan.csv')
# ==========================================================================================================================#
# ==========================================================================================================================#
# ============================================features after the loan time=================================================#
# ==========================================================================================================================#
# ==========================================================================================================================#
billDetailAfterLoan = billDetail2[billDetail2['billTimeStmp'] > billDetail2['loanTime']].drop('loanTime',axis =1)
billDetailGpAfterLoan = billDetailAfterLoan.groupby('userId')
# billFeatureAfterLoan = billDetailGpAfterLoan[['minPayThisMon','evolInst']].agg({'minMonthPaySum':'sum','evolInstSum':'sum'})
billFeatureAfterLoan = billDetailGpAfterLoan['minPayThisMon'].agg({'minMonthPaySum':'sum'}).reset_index()
billFeatureAfterLoan =pd.merge(billFeatureAfterLoan, billDetailGpAfterLoan['evolInst'].agg({'evolInstSum':'sum'}).reset_index(), how = 'outer', on = 'userId')
# billDescribeAfterLoan = billDetailGpAfterLoan.describe()
# billDescribeAfterLoan.to_csv('../describes/billDescribeAfterLoan.csv')
billDescribeAfterLoan = pd.read_csv('../describes/billDescribeAfterLoan.csv')
billDescribeAfterLoan = doubleIndex2single(billDescribeAfterLoan)
billFeatureAfterLoan = pd.merge(billFeatureAfterLoan,billDescribeAfterLoan, how = 'outer', on = 'userId')
colNamesAfterLoan = ['userId']
colNamesAfterLoan += (list(billFeatureAfterLoan.columns[1:]+ 'AfterLoan'))
billFeatureAfterLoan.columns = colNamesAfterLoan

billDetailFeatures = pd.merge(billFeatureBeforeLoan, billFeatureAfterLoan, how = 'outer', on = 'userId')
billDetailFeatures.to_csv('../featureFolderTrain/billDetailFeatures.csv')
