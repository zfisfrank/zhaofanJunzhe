import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()

from learning1 import DataCastle
sol = DataCastle()
user_info = sol.readUserInfo()
bank_detail = sol.readBankDetail()
bill_detail = sol.readBillDetail()
loan_time = sol.readLoanTime()
browse_history = sol.readBrowseHistory()
target = sol.readTarget()

loan_data = user_info.join(bank_detail,how='outer')
loan_data = loan_data.join(bill_detail,how='outer')
loan_data = loan_data.join(browse_history,how='outer')
loan_data = loan_data.join(loan_time,how='outer')
loan_data = loan_data.fillna(0.0)

#对数据进行归一化
datas = loan_data.values
datas = preprocessing.scale(datas)
col_names = list(loan_data.columns)
nums=0
for col in col_names:
    loan_data.loc[:,[col]] = datas[:,nums]
    nums += 1

#对数据进行划分并且进行训练
train = loan_data.iloc[0: 55596, :]
test = loan_data.iloc[55596:, :]
train_X, test_X, train_y, test_y = train_test_split(train,target,test_size = 0.2,random_state = 0)
train_y = train_y['label']
test_y = test_y['label']
lr_model = LogisticRegression(C = 1.0,penalty = 'l2')
lr_model.fit(train_X, train_y)
#验证集进行预测
pred_test = lr_model.predict(test_X)
#对预测结果进行评估
print(classification_report(test_y, pred_test))
sum(pred_test[test_y == 1] == 1)/len(pred_test)

ks(test_y, pred_test)
#对测试集生成结果并存储为csv格式
pred = lr_model.predict_proba(test)
result = pd.DataFrame(pred)
result.index = test.index
result.columns = ['0', 'probability']
result.drop('0',axis = 1,inplace = True)
print(result.head(5))
    # result.to_csv(sol.result)