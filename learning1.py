import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class DataCastle(object):
    def __init__(self):
        self.name = "<<- User loan forecast match ->>"
        self.result = "result.csv"

    #读取用户信息表 并返回
    def readUserInfo(self):
        user_info_train = readData("train/user_info_train.txt")
        user_info_test = readData("test/user_info_test.txt")
        col_names = ['userid', 'sex', 'occupation', 'education', 'marriage', 'household']
        user_info_train.columns = col_names
        user_info_test.columns = col_names
        user_info = pd.concat([user_info_train, user_info_test])
        user_info.index = user_info['userid']
        user_info.drop('userid',axis=1,inplace=True)
        return user_info

    #读取用户银行账单表 对账单数据求和并返回
    def readBankDetail(self):
        bank_detail_train = readData("train/bank_detail_train.txt")
        bank_detail_test = readData("test/bank_detail_test.txt")
        col_names = ['userid', 'time_bank', 'tradeType', 'tradeMoney', 'incomeTag']
        bank_detail_train.columns = col_names
        bank_detail_test.columns = col_names
        bank_detail_pre = pd.concat([bank_detail_train,bank_detail_test])
        bank_detail = (bank_detail_pre.loc[:,['userid','tradeType', 'tradeMoney']]).groupby(['userid','tradeType']).sum()
        bank_detail = bank_detail.unstack()
        bank_detail.columns = ['income','outcome']
        return bank_detail

    #读取用户的浏览历史 对浏览数据求和并返回
    def readBrowseHistory(self):
        browse_history_train = readData("train/browse_history_train.txt")
        browse_history_test = readData("test/browse_history_test.txt")
        col_names = ['userid', 'time_browse', 'browseData', 'browseTag']
        browse_history_train.columns = col_names
        browse_history_test.columns = col_names
        browse_history_pre = pd.concat([browse_history_train, browse_history_test])
        browse_history = (browse_history_pre.loc[:,['userid','browseData']]).groupby(['userid']).sum()
        return browse_history

    #读取信用卡账单记录 取均值并返回
    def readBillDetail(self):
        bill_detail_train = readData("train/bill_detail_train.txt")
        bill_detail_test = readData("test/bill_detail_test.txt")
        col_names = ['userid', 'time_bill', 'bank_id', 'prior_account', 'prior_repay',
             'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
             'account', 'adjust_account', 'circulated_interest', 'avaliable_balance',
             'cash_limit', 'repay_state']
        bill_detail_train.columns = col_names
        bill_detail_test.columns = col_names
        bill_detail_pre = pd.concat([bill_detail_train,bill_detail_test])
        bill_detail_pre.drop('bank_id',axis=1,inplace=True)
        bill_detail = bill_detail_pre.groupby(['userid']).mean()
        return bill_detail

    #读取用户发放贷款时间 并返回
    def readLoanTime(self):
        loan_time_train = readData("train/loan_time_train.txt")
        loan_time_test = readData("test/loan_time_test.txt")
        col_names = ['userid','loanTime']
        loan_time_train.columns = col_names
        loan_time_test.columns = col_names
        loan_time = pd.concat([loan_time_train,loan_time_test])
        loan_time.index = loan_time['userid']
        loan_time.drop('userid',axis=1,inplace=True)
        return loan_time

     #读取类别信息
    def readTarget(self):
        target = readData("train/overdue_train.txt")
        target.columns = ['userid', 'label']
        target.index = target['userid']
        target.drop('userid',axis = 1,inplace = True)
        return target

    #利用逻辑斯蒂回归
    def logisticMethod(self):

        user_info = self.readUserInfo()
        bank_detail = self.readBankDetail()
        bill_detail = self.readBillDetail()
        loan_time = self.readLoanTime()
        browse_history = self.readBrowseHistory()
        target = self.readTarget()

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

        #对测试集生成结果并存储为csv格式
        pred = lr_model.predict_proba(test)
        result = pd.DataFrame(pred)
        result.index = test.index
        result.columns = ['0', 'probability']
        result.drop('0',axis = 1,inplace = True)
        print(result.head(5))
        result.to_csv(self.result)

#数据读取
def readData(filename):
    filepath = './'+filename
    data = pd.read_csv(filepath,header=None)
    return data