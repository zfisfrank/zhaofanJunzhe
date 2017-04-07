#usr/bin/python3
# 提取了 bank_detail_train, put information into one row per userId
import numpy as np
import pandas as pd



# 银行流水记录：用户id,时间戳,交易类型,交易金额,工资收入标记
bankDetailCol = ['userId','timeStamp','transType','transAmount','salaryIncome']
bankDetail = pd.read_csv('bank_detail_train.csv',index_col = False, header  = None, names = bankDetailCol)
# 信用卡账单记录：用户id，账单时间戳，银行id，上期账单金额， 上期还款金额，信用卡额度，本期账单余额，
# 本期账单最低还款，消费笔数,本期账单金额，调整金额，循环利息，可用余额，预借现金额度， 还款状态
# possible noise : bankId
# 加入 用户使用年数； grouping by time range rows of bankDetail, billDetail of same user ;
billDetailCol = \
    ['userId','billTimeStmp','bankId','lastBillAmt','lastPaidAmt','creditAmount','remainedBalThisMon',\
    'minPayThisMon','#ofTrans','balThisMon','ajtedAmt','evolInst','remainBal','cashCredictLimit','payStus']
billDetail = pd.read_csv('bill_detail_train.csv',index_col = False, header  = None, names = billDetailCol)
 # 用户浏览行为: 用户id,时间戳,浏览行为数据,浏览子行为编号
browseHistCol = ['userId','timeStmp','browsAct','browsId']
browseHist = pd.read_csv('browse_history_train.csv',index_col = False, header  = None, names = browseHistCol)




# from bankDetail import change_value
# bD = change_value(bankDetail,0,1,'transType')
# bD = change_value(bD,1,-1,'transType')

userInfo = pd.merge(how='inner',on = "userId")
