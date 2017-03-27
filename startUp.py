#usr/bin/python3

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
# 银行流水记录：用户id,时间戳,交易类型,交易金额,工资收入标记
bankDetailCol = ['userId','timeStamp','transType','transAmount','salaryIncome']
bankDetail = pd.read_csv('bank_detail_train.csv',index_col = False, header  = None, names = bandDetailCol)
# 信用卡账单记录：用户id，账单时间戳，银行id，上期账单金额， 上期还款金额，信用卡额度，本期账单余额，
# 本期账单最低还款，消费笔数,本期账单金额，调整金额，循环利息，可用余额，预借现金额度， 还款状态
billDetailCol = \
    ['userId','billTimeStmp','bankId','lastBillAmt','lastPaidAmt','creditAmount','remainedBalThisMon',\
    'minPayThisMon','#ofTrans','balThisMon','ajtedAmt','evolInst','remainBal','cashedAmt','payStus']
billDetail = pd.read_csv('bill_detail_train.csv',index_col = False, header  = None, names = billDetailCol)
 # 用户浏览行为: 用户id,时间戳,浏览行为数据,浏览子行为编号
browseHistCol = ['userId','timeStmp','browsAct','browsId']
browseHist = pd.read_csv('browse_history_train.csv',index_col = False, header  = None, names = browseHistCol)
# 放款时间信息: 用户id,放款时间
loanTimeCol = ['userId','loanTime']
loanTime = pd.read_csv('loan_time_train.csv',index_col = False, header  = None, names = loanTimeCol)

# 逾期行为的记录: 用户id,样本标签
overDueCol = ['userId','label']
overDue = pd.read_csv('overdue_train.csv',index_col = False, header  = None)
