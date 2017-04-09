billDetailGpAfterLoan = billDetailAfterLoan.groupby('userId')
billFeatureAfterLoan = billDetailGpAfterLoan[['minPayThisMon','evolInst']].agg({'minMonthPaySum':'sum','evolInstSum':'sum'})
billFeatureAfterLoan = billFeatureAfterLoan.reset_index()
billDescribeAfterLoan = billDetailGpAfterLoan.describe()
