# /usr/local/bin/python3

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# normalize to 0~1
def normalize(df, despiteCol = []):
    interstedCol = df.columns[~df.columns.isin(despiteCol)]
    result = (df[interstedCol] - df[interstedCol].min())/(df[interstedCol].max() - df[interstedCol].min())
    # df = (df - df.min())/(df.max() - df.min())
    result = result.dropna(how ='all',axis = 1)
    return result

# calculate all pearson correlation coefficient of columns in dataFrame with y(target)
def pearCorr(df,y):
    result =  df.apply(lambda x: pearsonr(x,y)).apply(pd.Series)
    result.set_axis(1,['Pearson’s correlation coefficient','2-tailed p-value'])
    return result
