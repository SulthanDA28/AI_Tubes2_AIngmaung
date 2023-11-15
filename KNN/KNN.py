import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import jit
data = pd.read_csv('./data_train.csv')
validation = pd.read_csv('./data_validation.csv')

# print(row["blue"])
# print(row)
def normalisasi(data):
    for col in data.columns:
        if col != 'price_range':
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data

data = normalisasi(data)
validation = normalisasi(validation)


def knn(data, k, cari):
    result = pd.DataFrame(columns=['distance', 'price_range'], index=range(data.shape[0]))
    for i in range(data.shape[0]):
        jml = 0
        for col in ['ram','px_width', 'battery_power','px_height']:
            if col != 'price_range':
                jml += (data[col].iloc[i] - cari[col]) ** 2
        result['distance'].iloc[i] = np.sqrt(jml)
        result['price_range'].iloc[i]= data.iloc[i]['price_range']
    result = result.sort_values(by=['distance'])
    result = result.iloc[0:k]
    hasil = result.groupby('price_range').count().idxmax()['distance']
    return hasil

# print(validation["price_range"].iloc[1])
# knn(data, 5, validation.iloc[1])

truePositive = 0
trueNegative = 0
sqrtn = int(np.sqrt(data.shape[0])) 
for i in tqdm(range(validation.shape[0])):
    if validation["price_range"].iloc[i] == knn(data, sqrtn, validation.iloc[i]):
        truePositive += 1
    else:
        trueNegative += 1
        
print(truePositive/(truePositive+trueNegative)*100)
