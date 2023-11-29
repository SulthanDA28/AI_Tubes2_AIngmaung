import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

class KNNclassifier:
    def __init__(self, training_data : pd.DataFrame, k : int, x_labels : list[str], weights : list[float], y_label : str):
        self.training_data = training_data
        self.k = k
        self.x_labels = x_labels
        self.y_label = y_label
        self.weights = weights

    def predict(self, test_data : pd.DataFrame) -> pd.DataFrame:
        self._normalize(self.training_data, test_data)
        result = pd.DataFrame(columns=['id', 'price_range'], index=range(test_data.shape[0]))
        for i in range(test_data.shape[0]):
            result['id'].iloc[i] = i
            result['price_range'].iloc[i] = self.predict_one(test_data.iloc[i])
        return result

    def predict_one(self, sample : pd.DataFrame) -> int:
        result = pd.DataFrame(columns=['distance', self.y_label], index=range(self.training_data.shape[0]))
        eucDist = np.sqrt(np.sum(self.weights * (self.training_data[self.x_labels].values - sample[self.x_labels].values)**2, axis=1))
        result['distance'] = eucDist
        result[self.y_label] = self.training_data[self.y_label]
        result = result.sort_values(by=['distance'])
        result = result.iloc[0:self.k]
        score = np.zeros(4)
        for i in range(self.k):
            if result['distance'].iloc[i] == 0:
                return result[self.y_label].iloc[i]
            score[result[self.y_label].iloc[i]] += 1 / (result['distance'].iloc[i])
        return np.argmax(score)

    def _normalize(self, train_data : pd.DataFrame, test_data : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        for col in train_data.columns:
            if col != 'price_range':
                MIN = min(train_data[col].min(),test_data[col].min())
                MAX = max(train_data[col].max(),test_data[col].max())
                train_data[col] = (train_data[col] - MIN) / (MAX - MIN)
                test_data[col] = (test_data[col] - MIN) / (MAX - MIN)
        return train_data, test_data

if __name__ == "__main__":
    train_data = pd.read_csv("Data/" + sys.argv[1])
    test_data = pd.read_csv("Data/" + sys.argv[2])
    K = 13
    classifier = KNNclassifier(train_data, K, ['ram','px_width', 'battery_power', 'px_height'], [0.7,0.1,0.1,0.1], 'price_range')
    result = classifier.predict(test_data)
    result.to_csv("Out/" + sys.argv[3], index=False)
    print("Result saved to Out/" + sys.argv[3])
