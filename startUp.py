#usr/bin/python3

import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
# import string
from joblib import Parallel, delayed
from sklearn import preprocessing

bankDetail = pd.read_csv('bank_detail_train.csv')
billDetail = pd.read_csv('bill_detail_train.csv')
browseHist = pd.read_csv('browse_history_train.csv')
loanTime = pd.read_csv('loan_time_train.csv')
overDue = pd.read_csv('overdue_train.csv')
userInfo = pd.read_csv('user_info_train.csv')
