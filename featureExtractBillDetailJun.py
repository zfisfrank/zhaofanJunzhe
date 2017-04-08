#usr/bin/python3


# 提取了 billdetail, put information into one row per userId
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

userInfo = pd.read_csv('../dataSets/userInfoTrain.csv').sort_values('userId')
billDetail = pd.read_csv('../dataSets/billDetailTrain.csv')
billDetail2 = pd.merge(billDetail,userInfo[['userId','loanTime']],how = 'right',on = 'userId')
# ==========================================================================================================================#
# ==========================================================================================================================#
# ============================================features before the loan time=================================================#
# ==========================================================================================================================#
# ==========================================================================================================================#

billDetailBeforeLoan = billDetail2[billDetail2['timeStmp'] <= billDetail2['loanTime']].drop('loanTime',axis =1)
billDetailGpBeforeLoan = billDetailBeforeLoan.groupby('userId')
#describe on hold first
# billDetailDescribeBeforeLoan = billDetailGpBeforeLoan.describe()
# as the cal time is very slow, save the result into a file for later use
# billDetailDescribeBeforeLoan.to_csv('../dataSets/billDetailDescribeBeforeLoan.csv')
# billDetailDescribe = billDetailDescribe.reset_index()

# just read pre-processed 'billDetailDescribe'
# billDetailDescribe = pd.read_csv('../dataSets/billDetailDescribeBeforeLoan.csv')

#get sum of each column
billDetailGpBeforeLoanSum = billDetailGpBeforeLoan.sum()
billDetailGpBeforeLoanSum.columns = ['userId', 'BLbillTimeStmp', 'bankId', 'lastBillAmt', 'lastPaidAmt',
'creditAmount', 'remainedBalThisMon', 'minPayThisMon', '#ofTrans', 'balThisMon', 'ajtedAmt',
 'evolInst', 'remainBal', 'cashCredictLimit', 'payStus']
