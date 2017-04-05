##usr/bin/python3

import pandas as pd

# if cell of dataFrame[index] == changeVal, change it to changeTo
def change_value(dF, toReplace, changeTo, index):
    interetedSeries = dF[index]
    interetedSeries = interetedSeries.replace(toReplace,changeTo)
    dF[index] = interetedSeries
    return dF

# modify bankDetail to one userId contains one row
def modify_bank_detail(dF):
    dF = change_value(bankDetail,0,1,'transType')
    dF = change_value(bD,1,-1,'transType')
    return dF
