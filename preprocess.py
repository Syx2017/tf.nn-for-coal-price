from __future__ import print_function
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
pd.options.display.max_columns = 10
pd.options.display.max_rows = 10

def dataprepare(filename, time, feature, dataframe):
    ori_dataframe = pd.read_csv(filename)
    valuelist = []
    res = []
    month = 1
    for i in range(ori_dataframe[time].count()):
        timelist = ori_dataframe[time][i].split('/')
        if int(timelist[1]) == month:
            if ori_dataframe[feature][i] != 0:
                valuelist.append(ori_dataframe[feature][i])
        else:
            res.append(sum(valuelist) / len(valuelist))
            valuelist.clear()
            valuelist.append(ori_dataframe[feature][i])
            if int(timelist[1]) == month + 1:
                month += 1
            else:
                month = 1
    # standarlize
    # data standarize

    if feature != 'CCTDQH5500':
        res = np.array(res).reshape(-1, 1)
        x_scaler = StandardScaler()
        x_scaler.fit(res)
        X_standarized = x_scaler.transform(res).reshape(-1, )
        dataframe[feature] = np.array(X_standarized)
        return
    else:
        dataframe[feature] = res
        return
#print(dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv','day','CCI5000'))




