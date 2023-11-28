import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

data = pd.read_csv('./data_train.csv')
validation = pd.read_csv('./data_validation.csv')
features = ['ram','px_width', 'battery_power', 'px_height']
x = data[['ram','px_width', 'battery_power', 'px_height']].values
y = data.iloc[:, -1].values

x_val = validation[['ram','px_width', 'battery_power', 'px_height']].values
y_val = validation.iloc[:, -1].values

scaler = preprocessing.MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_val = scaler.transform(x_val)
print(x)
print(x_val)

knn = KNeighborsClassifier(n_neighbors=33)
knn.fit(x, y)
y_pred = knn.predict(x_val)

print(accuracy_score(y_val, y_pred))

