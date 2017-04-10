# python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 用户id,时间戳,交易类型,交易金额,工资收入标记
bankDetailCol = ['userId','timeStamp','transType','transAmount','salaryIncome']
bankDetail = pd.read_csv('../initData/bank_detail_train.txt',index_col =False,
    header  = None, names = bankDetailCol).sort_values('userId')
# bankDetail = bankDetail.reset_index()
# bankDetail group
# bankDetailGp = bankDetail.groupby('userId')

bankDetailTotalIncome = bankDetail[bankDetail['transType'] == 0].groupby('userId').agg({'incomeCountBeforeLoan':'count','totalIncomeBeforeLoan':'sum'})
