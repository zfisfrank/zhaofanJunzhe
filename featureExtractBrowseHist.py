#usr/bin/python3
# need to run after featureExtractBoankDetail.py

# 提取了 browseHist, put information into one row per userId
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# need to use a method to reconstruct the information table
# input doubleDf need to have the reset_index() result
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
browseHist = pd.read_csv('../dataSets/browseHistTrain.csv')
browseHist2 = pd.merge(browseHist,userInfo[['userId','loanTime']],how = 'right',on = 'userId')
# ==========================================================================================================================#
# ==========================================================================================================================#
# ============================================features before the loan time=================================================#
# ==========================================================================================================================#
# ==========================================================================================================================#
browseHistBeforeLoan = browseHist2[browseHist2['timeStmp'] <= browseHist2['loanTime']].drop('loanTime',axis =1)

brewseHistGpBeforeLoan = browseHistBeforeLoan.groupby('userId')
# brewseHistDescribeBeforeLoan = brewseHistGpBeforeLoan.describe()
# as the cal time is very slow, save the result into a file for later use
# brewseHistDescribeBeforeLoan.to_csv('../describes/brewseHistDescribeBeforeLoan.csv')
# brewseHistDescribeBeforeLoan = brewseHistDescribeBeforeLoan.reset_index()

# just read pre-processed 'brewseHistDescribe'
brewseHistDescribeBeforeLoan = pd.read_csv('../describes/brewseHistDescribeBeforeLoan.csv')
 # change time stamp into hours, so showing hour information
brewseHistDescribeBeforeLoan['timeStmp'] = (brewseHistDescribeBeforeLoan['timeStmp']//3600)%24

# re-construct table to one userId contains only one row
brewseHistDescribeBeforeLoan = doubleIndex2single(brewseHistDescribeBeforeLoan)
colNamesBeforeLoan = ['userId']
colNamesBeforeLoan += (list(brewseHistDescribeBeforeLoan.columns[1:]+ 'BeforeLoan'))
brewseHistDescribeBeforeLoan.columns = colNamesBeforeLoan
# brewseHistDescribeBeforeLoan.columns[1:] = brewseHistDescribeBeforeLoan.columns[1:]+ 'BeforeLoan'
# brewseHistBeforeLoan = brewseHistGpBeforeLoan['browsId'].agg({'browsIdCount':'count'})


# ==========================================================================================================================#
# ==========================================================================================================================#
# ============================================features after the loan time=================================================#
# ==========================================================================================================================#
# ==========================================================================================================================#
# After loan time
browseHistAfterLoan = browseHist2[browseHist2['timeStmp'] > browseHist2['loanTime']].drop('loanTime',axis =1)

brewseHistGpAfterLoan = browseHistAfterLoan.groupby('userId')
brewseHistDescribeAfterLoan = brewseHistGpAfterLoan.describe()
# as the cal time is very slow, save the result into a file for later use
brewseHistDescribeAfterLoan.to_csv('../describes/brewseHistDescribeAfterLoan.csv')
brewseHistDescribeAfterLoan = brewseHistDescribeAfterLoan.reset_index()

# just read pre-processed 'brewseHistDescribe'
brewseHistDescribeAfterLoan = pd.read_csv('../describes/brewseHistDescribeAfterLoan.csv')
 # change time stamp into hours, so showing hour information
brewseHistDescribeAfterLoan['timeStmp'] = (brewseHistDescribeAfterLoan['timeStmp']//3600)%24

# re-construct table to one userId contains only one row
brewseHistDescribeAfterLoan = doubleIndex2single(brewseHistDescribeAfterLoan)
colNamesAfterLoan = ['userId']
colNamesAfterLoan += (list(brewseHistDescribeAfterLoan.columns[1:]+ 'AfterLoan'))
brewseHistDescribeAfterLoan.columns = colNamesAfterLoan

features = pd.merge(brewseHistDescribeBeforeLoan, brewseHistDescribeAfterLoan, how = 'outer', on = 'userId')

features.to_csv('../dataSets/browseHistFeaturesTrain.csv')
