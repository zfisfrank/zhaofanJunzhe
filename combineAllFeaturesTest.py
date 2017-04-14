import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from modifyFun import combineAllInfo
from modifyFun import ks
from modifyFun import readUserInfo

userInfo, label = readUserInfo()
allData,labels  = combineAllInfo()
# histFeatures = pd.read_csv('../feature/browseHistFeaturesFull.csv').set_index('userId')
# histFeatures = features.set_index('userId')
allData = pd.concat([allData,histFeatures],axis = 1)

bankDetail = pd.read_csv('../feature/bankDetailsFeaturesFull.csv').set_index('userId')
billDetail = pd.read_csv('../feature/billDetailFeaturesFull.csv').set_index('userId')


# billDetailFeaturesTrain = billDetail
# ============================================================================================================
# ============================================================================================================
# ============================================================================================================

# jun zhe's job

bankDetail = pd.read_csv('../feature/bankDetailsFeaturesFull.csv').set_index('userId')
billDetail = pd.read_csv('../feature/billDetailFeaturesFull.csv').set_index('userId')
browseHist = pd.read_csv('../feature/browseHistFeaturesFull.csv').set_index('userId')
browHist = pd.read_csv('../feature/browseHistFeaturesFullDense.csv').set_index('userId')
# colE=list(billDetailFeaturesTrain.columns.values)
colE=list(billDetail.columns.values)
#delta 'lastPaidAmt','lastBillAmt'
#delta 'creditAmount', 'remainedBalThisMon'
#delta 'remainBal', 'cashCredictLimit'
#sum minPayThisMon' , 'evolInst'
lastPaidAmt= [i for i in colE if  'lastPaidAmt' in i]
lastBillAmt=[i for i in colE if 'lastBillAmt' in i]
creditAmount=[i for i in colE if  'creditAmount' in i]
remainedBalThisMon=[i for i in colE if 'remainedBalThisMon' in i]
CremainedBalThisMon=[i for i in colE if  'remainBal' in  i]
CremainedBalThisMon=[i for i in CremainedBalThisMon if not 'Vscash' in i] # ewmove Vscash
cashCredictLimit=[i for i in colE if  'cashCredictLimit' in i]
minPayThisMon=[i for i in colE if  'minPayThisMon' in i]
evolInst=[i for i in colE if  'evolInst' in i]
evolInst=[i for i in evolInst if '_' in i] #remove without"_"

# print ("length of Sub")
# print (len(lastPaidAmt))
# print (len(lastBillAmt))
# #print lastPaidAmt
# #print lastBillAmt
#
# print len(creditAmount)
# print len(remainedBalThisMon)
# #print creditAmount
# #print remainedBalThisMon
#
# print len(CremainedBalThisMon)
# print len(cashCredictLimit)
# #print CremainedBalThisMon
# #print cashCredictLimit
#
# print len(minPayThisMon)
# print len(evolInst)a
#print minPayThisMon
#print evolInst

list1=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16']
list2 = [word.replace('a','b') for word in list1]
list3 = [word.replace('a','c') for word in list1]
list4 = [word.replace('a','d') for word in list1]
#print list1+list2+list3+list4


allData2 = pd.concat([userInfo,bankDetail,billDetail,browseHist],axis = 1)
# allData2 = pd.concat([bankDetail,billDetail,browseHist],axis = 1)

allData2[list1] = allData2[lastPaidAmt] - allData2[lastBillAmt].values
allData2[list2] = allData2[creditAmount] - allData2[remainedBalThisMon].values
allData2[list3] = allData2[CremainedBalThisMon] - allData2[cashCredictLimit].values
allData2[list4] = allData2[minPayThisMon] + allData2[evolInst].values

colSub=list1+list2+list3+list4
print (len(colSub))

interstCol = ['userId', 'overDueLabel', 'gender_0', 'gender_1', 'gender_2', 'job_0', 'job_1', 'job_2', 'job_3', 'job_4', 'education_0', 'education_1', 'education_2', 'education_3', 'education_4', 'marriage_0', 'marriage_1', 'marriage_2', 'marriage_3', 'marriage_4', 'marriage_5', 'residentialType_0', 'residentialType_1', 'residentialType_2', 'residentialType_3', 'residentialType_4', 'firstIncomeDayBeforeLoan', 'lastIncomeDayBeforeLoan', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'userId', 'minMonthPaySumBeforeLoan', 'evolInstSumBeforeLoan', 'count_#ofTransBeforeLoan', 'count_ajtedAmtBeforeLoan', 'count_balThisMonBeforeLoan', 'count_bankIdBeforeLoan', 'count_billTimeStmpBeforeLoan', 'count_cashCredictLimitBeforeLoan', 'count_creditAmountBeforeLoan', 'count_evolInstBeforeLoan', 'count_lastBillAmtBeforeLoan', 'count_lastPaidAmtBeforeLoan', 'count_minPayThisMonBeforeLoan', 'count_payStusBeforeLoan', 'count_remainBalBeforeLoan', 'count_remainBalVscashCreditLimitBeforeLoan', 'count_remainedBalThisMonBeforeLoan', 'count_spentCreditThisMonthBeforeLoan', 'count_unPaidedBillBeforeLoan', 'mean_#ofTransBeforeLoan', 'mean_ajtedAmtBeforeLoan', 'mean_balThisMonBeforeLoan', 'mean_bankIdBeforeLoan', 'mean_billTimeStmpBeforeLoan', 'mean_cashCredictLimitBeforeLoan', 'mean_creditAmountBeforeLoan', 'mean_evolInstBeforeLoan', 'mean_lastBillAmtBeforeLoan', 'mean_lastPaidAmtBeforeLoan', 'mean_minPayThisMonBeforeLoan', 'mean_payStusBeforeLoan', 'mean_remainBalBeforeLoan', 'mean_remainBalVscashCreditLimitBeforeLoan', 'mean_remainedBalThisMonBeforeLoan', 'mean_spentCreditThisMonthBeforeLoan', 'mean_unPaidedBillBeforeLoan', 'std_#ofTransBeforeLoan', 'std_ajtedAmtBeforeLoan', 'std_balThisMonBeforeLoan', 'std_bankIdBeforeLoan', 'std_billTimeStmpBeforeLoan', 'std_cashCredictLimitBeforeLoan', 'std_creditAmountBeforeLoan', 'std_evolInstBeforeLoan', 'std_lastBillAmtBeforeLoan', 'std_lastPaidAmtBeforeLoan', 'std_minPayThisMonBeforeLoan', 'std_payStusBeforeLoan', 'std_remainBalBeforeLoan', 'std_remainBalVscashCreditLimitBeforeLoan', 'std_remainedBalThisMonBeforeLoan', 'std_spentCreditThisMonthBeforeLoan', 'std_unPaidedBillBeforeLoan', 'min_#ofTransBeforeLoan', 'min_ajtedAmtBeforeLoan', 'min_balThisMonBeforeLoan', 'min_bankIdBeforeLoan', 'min_billTimeStmpBeforeLoan', 'min_cashCredictLimitBeforeLoan', 'min_creditAmountBeforeLoan', 'min_evolInstBeforeLoan', 'min_lastBillAmtBeforeLoan', 'min_lastPaidAmtBeforeLoan', 'min_minPayThisMonBeforeLoan', 'min_payStusBeforeLoan', 'min_remainBalBeforeLoan', 'min_remainBalVscashCreditLimitBeforeLoan', 'min_remainedBalThisMonBeforeLoan', 'min_spentCreditThisMonthBeforeLoan', 'min_unPaidedBillBeforeLoan', '25%_#ofTransBeforeLoan', '25%_ajtedAmtBeforeLoan', '25%_balThisMonBeforeLoan', '25%_bankIdBeforeLoan', '25%_billTimeStmpBeforeLoan', '25%_cashCredictLimitBeforeLoan', '25%_creditAmountBeforeLoan', '25%_evolInstBeforeLoan', '25%_lastBillAmtBeforeLoan', '25%_lastPaidAmtBeforeLoan', '25%_minPayThisMonBeforeLoan', '25%_payStusBeforeLoan', '25%_remainBalBeforeLoan', '25%_remainBalVscashCreditLimitBeforeLoan', '25%_remainedBalThisMonBeforeLoan', '25%_spentCreditThisMonthBeforeLoan', '25%_unPaidedBillBeforeLoan', '50%_#ofTransBeforeLoan', '50%_ajtedAmtBeforeLoan', '50%_balThisMonBeforeLoan', '50%_bankIdBeforeLoan', '50%_billTimeStmpBeforeLoan', '50%_cashCredictLimitBeforeLoan', '50%_creditAmountBeforeLoan', '50%_evolInstBeforeLoan', '50%_lastBillAmtBeforeLoan', '50%_lastPaidAmtBeforeLoan', '50%_minPayThisMonBeforeLoan', '50%_payStusBeforeLoan', '50%_remainBalBeforeLoan', '50%_remainBalVscashCreditLimitBeforeLoan', '50%_remainedBalThisMonBeforeLoan', '50%_spentCreditThisMonthBeforeLoan', '50%_unPaidedBillBeforeLoan', '75%_#ofTransBeforeLoan', '75%_ajtedAmtBeforeLoan', '75%_balThisMonBeforeLoan', '75%_bankIdBeforeLoan', '75%_billTimeStmpBeforeLoan', '75%_cashCredictLimitBeforeLoan', '75%_creditAmountBeforeLoan', '75%_evolInstBeforeLoan', '75%_lastBillAmtBeforeLoan', '75%_lastPaidAmtBeforeLoan', '75%_minPayThisMonBeforeLoan', '75%_payStusBeforeLoan', '75%_remainBalBeforeLoan', '75%_remainBalVscashCreditLimitBeforeLoan', '75%_remainedBalThisMonBeforeLoan', '75%_spentCreditThisMonthBeforeLoan', '75%_unPaidedBillBeforeLoan', 'max_#ofTransBeforeLoan', 'max_ajtedAmtBeforeLoan', 'max_balThisMonBeforeLoan', 'max_bankIdBeforeLoan', 'max_billTimeStmpBeforeLoan', 'max_cashCredictLimitBeforeLoan', 'max_creditAmountBeforeLoan', 'max_evolInstBeforeLoan', 'max_lastBillAmtBeforeLoan', 'max_lastPaidAmtBeforeLoan', 'max_minPayThisMonBeforeLoan', 'max_payStusBeforeLoan', 'max_remainBalBeforeLoan', 'max_remainBalVscashCreditLimitBeforeLoan', 'max_remainedBalThisMonBeforeLoan', 'max_spentCreditThisMonthBeforeLoan', 'max_unPaidedBillBeforeLoan', 'minMonthPaySumAfterLoan', 'evolInstSumAfterLoan', 'count_#ofTransAfterLoan', 'count_ajtedAmtAfterLoan', 'count_balThisMonAfterLoan', 'count_bankIdAfterLoan', 'count_billTimeStmpAfterLoan', 'count_cashCredictLimitAfterLoan', 'count_creditAmountAfterLoan', 'count_evolInstAfterLoan', 'count_lastBillAmtAfterLoan', 'count_lastPaidAmtAfterLoan', 'count_minPayThisMonAfterLoan', 'count_payStusAfterLoan', 'count_remainBalAfterLoan', 'count_remainBalVscashCreditLimitAfterLoan', 'count_remainedBalThisMonAfterLoan', 'count_spentCreditThisMonthAfterLoan', 'count_unPaidedBillAfterLoan', 'mean_#ofTransAfterLoan', 'mean_ajtedAmtAfterLoan', 'mean_balThisMonAfterLoan', 'mean_bankIdAfterLoan', 'mean_billTimeStmpAfterLoan', 'mean_cashCredictLimitAfterLoan', 'mean_creditAmountAfterLoan', 'mean_evolInstAfterLoan', 'mean_lastBillAmtAfterLoan', 'mean_lastPaidAmtAfterLoan', 'mean_minPayThisMonAfterLoan', 'mean_payStusAfterLoan', 'mean_remainBalAfterLoan', 'mean_remainBalVscashCreditLimitAfterLoan', 'mean_remainedBalThisMonAfterLoan', 'mean_spentCreditThisMonthAfterLoan', 'mean_unPaidedBillAfterLoan', 'std_#ofTransAfterLoan', 'std_ajtedAmtAfterLoan', 'std_balThisMonAfterLoan', 'std_bankIdAfterLoan', 'std_billTimeStmpAfterLoan', 'std_cashCredictLimitAfterLoan', 'std_creditAmountAfterLoan', 'std_evolInstAfterLoan', 'std_lastBillAmtAfterLoan', 'std_lastPaidAmtAfterLoan', 'std_minPayThisMonAfterLoan', 'std_payStusAfterLoan', 'std_remainBalAfterLoan', 'std_remainBalVscashCreditLimitAfterLoan', 'std_remainedBalThisMonAfterLoan', 'std_spentCreditThisMonthAfterLoan', 'std_unPaidedBillAfterLoan', 'min_#ofTransAfterLoan', 'min_ajtedAmtAfterLoan', 'min_balThisMonAfterLoan', 'min_bankIdAfterLoan', 'min_billTimeStmpAfterLoan', 'min_cashCredictLimitAfterLoan', 'min_creditAmountAfterLoan', 'min_evolInstAfterLoan', 'min_lastBillAmtAfterLoan', 'min_lastPaidAmtAfterLoan', 'min_minPayThisMonAfterLoan', 'min_payStusAfterLoan', 'min_remainBalAfterLoan', 'min_remainBalVscashCreditLimitAfterLoan', 'min_remainedBalThisMonAfterLoan', 'min_spentCreditThisMonthAfterLoan', 'min_unPaidedBillAfterLoan', '25%_#ofTransAfterLoan', '25%_ajtedAmtAfterLoan', '25%_balThisMonAfterLoan', '25%_bankIdAfterLoan', '25%_billTimeStmpAfterLoan', '25%_cashCredictLimitAfterLoan', '25%_creditAmountAfterLoan', '25%_evolInstAfterLoan', '25%_lastBillAmtAfterLoan', '25%_lastPaidAmtAfterLoan', '25%_minPayThisMonAfterLoan', '25%_payStusAfterLoan', '25%_remainBalAfterLoan', '25%_remainBalVscashCreditLimitAfterLoan', '25%_remainedBalThisMonAfterLoan', '25%_spentCreditThisMonthAfterLoan', '25%_unPaidedBillAfterLoan', '50%_#ofTransAfterLoan', '50%_ajtedAmtAfterLoan', '50%_balThisMonAfterLoan', '50%_bankIdAfterLoan', '50%_billTimeStmpAfterLoan', '50%_cashCredictLimitAfterLoan', '50%_creditAmountAfterLoan', '50%_evolInstAfterLoan', '50%_lastBillAmtAfterLoan', '50%_lastPaidAmtAfterLoan', '50%_minPayThisMonAfterLoan', '50%_payStusAfterLoan', '50%_remainBalAfterLoan', '50%_remainBalVscashCreditLimitAfterLoan', '50%_remainedBalThisMonAfterLoan', '50%_spentCreditThisMonthAfterLoan', '50%_unPaidedBillAfterLoan', '75%_#ofTransAfterLoan', '75%_ajtedAmtAfterLoan', '75%_balThisMonAfterLoan', '75%_bankIdAfterLoan', '75%_billTimeStmpAfterLoan', '75%_cashCredictLimitAfterLoan', '75%_creditAmountAfterLoan', '75%_evolInstAfterLoan', '75%_lastBillAmtAfterLoan', '75%_lastPaidAmtAfterLoan', '75%_minPayThisMonAfterLoan', '75%_payStusAfterLoan', '75%_remainBalAfterLoan', '75%_remainBalVscashCreditLimitAfterLoan', '75%_remainedBalThisMonAfterLoan', '75%_spentCreditThisMonthAfterLoan', '75%_unPaidedBillAfterLoan', 'max_#ofTransAfterLoan', 'max_ajtedAmtAfterLoan', 'max_balThisMonAfterLoan', 'max_bankIdAfterLoan', 'max_billTimeStmpAfterLoan', 'max_cashCredictLimitAfterLoan', 'max_creditAmountAfterLoan', 'max_evolInstAfterLoan', 'max_lastBillAmtAfterLoan', 'max_lastPaidAmtAfterLoan', 'max_minPayThisMonAfterLoan', 'max_payStusAfterLoan', 'max_remainBalAfterLoan', 'max_remainBalVscashCreditLimitAfterLoan', 'max_remainedBalThisMonAfterLoan', 'max_spentCreditThisMonthAfterLoan', 'max_unPaidedBillAfterLoan']
dropCol = ['count_billTimeStmpBeforeLoan', 'mean_billTimeStmpBeforeLoan', 'std_billTimeStmpBeforeLoan',
 'min_billTimeStmpBeforeLoan', '25%_billTimeStmpBeforeLoan', '50%_billTimeStmpBeforeLoan',
 '75%_billTimeStmpBeforeLoan', 'max_billTimeStmpBeforeLoan', 'count_billTimeStmpAfterLoan',
 'mean_billTimeStmpAfterLoan', 'std_billTimeStmpAfterLoan', 'min_billTimeStmpAfterLoan',
 '25%_billTimeStmpAfterLoan', '50%_billTimeStmpAfterLoan', '75%_billTimeStmpAfterLoan', 'max_billTimeStmpAfterLoan']

interstCol = [x for x in interstCol if x not in dropCol]
avaCol = list(allData2.columns)
interstCol = [x for x in interstCol if x in avaCol]
allData2 = allData2[interstCol]
allData2 = allData2.join(browHist,how = 'outer')
allData2 = allData2.T.drop_duplicates().T
# ============================================================================================================
# ============================================================================================================
# ============================================================================================================