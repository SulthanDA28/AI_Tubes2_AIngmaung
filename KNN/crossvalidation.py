import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from KNN import KNNclassifier

dataA = pd.read_csv("Data/data_train.csv")
dataB = pd.read_csv("Data/data_validation.csv")
combinedData = pd.concat([dataA,dataB])
print(combinedData.shape)

def calculate_accuracy(pred,true):
    true_positive = 0
    true_negative = 0
    for i in range(len(pred)):
        if pred['price_range'][i] == true['price_range'].iloc[pred['id'][i]]:
            true_positive += 1
        else:
            true_negative += 1

    return true_positive / (true_positive + true_negative)

kygmaudiuji = range(1,51)
groupnum = 10
sectionSize = combinedData.shape[0]//groupnum
resultTable = pd.DataFrame(index=kygmaudiuji,columns=["k"]+list(range(0,2000,sectionSize))+["avg","std"])
for k in tqdm(kygmaudiuji):
    for i in range(0,2000,sectionSize):
        validation = pd.DataFrame(combinedData[i:i+sectionSize]).reset_index(drop=True)
        data = pd.concat([combinedData[:i],combinedData[i+sectionSize:2000]]).reset_index(drop=True)
        classifier = KNNclassifier(data,k,['ram','px_width', 'battery_power', 'px_height'], [0.7,0.1,0.1,0.1], 'price_range')
        result = classifier.predict(validation)
        resultTable["k"][k] = k
        resultTable[i][k] = calculate_accuracy(result,validation)
    resultTable["avg"][k] = resultTable[list(range(0,2000,sectionSize))].iloc[k-1].mean()
    resultTable["std"][k] = resultTable[list(range(0,2000,sectionSize))].iloc[k-1].std()
resultTable.to_excel("Out/KNN-cross-validation-result.xlsx",index=False)
print("Result saved to Out/KNN-cross-validation-result.xlsx")
print(resultTable)