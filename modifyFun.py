# /usr/local/bin/python3

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# normalize to 0~1, first: dataframe to normalize, 2nd, ignore column name list
def normalize(df, despiteCol = ['userId']):
    interstedCol = df.columns[~df.columns.isin(despiteCol)]
    result = (df[interstedCol] - df[interstedCol].min())/(df[interstedCol].max() - df[interstedCol].min())
    result = pd.concat([df[despiteCol],result],axis = 1)
    # df = (df - df.min())/(df.max() - df.min())
    result = result.dropna(how ='all',axis = 1)
    return result

# normalize to number of std, first: dataframe to normalize, 2nd, ignore column name list
def stdNormalize(df, despiteCol = ['userId']):
    interstedCol = df.columns[~df.columns.isin(despiteCol)]
    # result = (df[interstedCol] - df[interstedCol].min())/(df[interstedCol].max() - df[interstedCol].min())
    result = (df[interstedCol] - df[interstedCol].mean())/df[interstedCol].std()
    result = pd.concat([df[despiteCol],result],axis = 1)
    # df = (df - df.min())/(df.max() - df.min())
    result = result.dropna(how ='all',axis = 1)
    return result

# calculate all pearson correlation coefficient of columns in dataFrame with y(target)
def pearCorr(df,y):
    result =  df.apply(lambda x: pearsonr(x,y)).apply(pd.Series)
    result.set_axis(1,['Pearson’s correlation coefficient','2-tailed p-value'])
    return result

from sklearn import metrics
def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()