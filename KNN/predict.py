import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

data = pd.read_csv(sys.argv[1])
validation = pd.read_csv(sys.argv[2])

def knn(data, k, cari, features, weights):
    result = pd.DataFrame(columns=['distance', 'price_range'], index=range(data.shape[0]))
    eucDist = np.sqrt(np.sum(weights * (data[features].values - cari[features].values)**2, axis=1))
    result['distance'] = eucDist
    result['price_range'] = data['price_range']
    result = result.sort_values(by=['distance'])
    result = result.iloc[0:k]
    score = np.zeros(4)
    for i in range(k):
        score[result['price_range'].iloc[i]] += 1 / (result['distance'].iloc[i])
    return np.argmax(score)

def normalization(data, validation):
    for col in data.columns:
        if col != 'price_range':
            MIN = min(data[col].min(),validation[col].min())
            MAX = max(data[col].max(),validation[col].max())
            data[col] = (data[col] - MIN) / (MAX - MIN)
            validation[col] = (validation[col] - MIN) / (MAX - MIN)
    return data, validation

normalization(data, validation)
# print(data[['ram','px_width', 'battery_power', 'px_height']])
# print(validation[['ram','px_width', 'battery_power', 'px_height']])
sqrtn = int(np.sqrt(data.shape[0]))

k = 15
result = pd.DataFrame(columns=['id', 'price_range'], index=range(validation.shape[0]))
for i in tqdm(range(validation.shape[0])):
    result['id'].iloc[i] = i
    result['price_range'].iloc[i] = knn(data, k, validation.iloc[i], 
                                        ['ram','px_width', 'battery_power', 'px_height'], 
                                        [0.7,0.1,0.1,0.1])
# print("accuracy: ", calculate_accuracy(result, validation))
result.to_csv("KNN/" + sys.argv[3], index=False)
print("result saved to KNN/" + sys.argv[3])
