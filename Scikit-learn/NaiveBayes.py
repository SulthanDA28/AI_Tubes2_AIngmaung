import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('./Data/data_train.csv')
validation = pd.read_csv('./Data/data_validation.csv')
feature1 = ['ram','px_width', 'battery_power', 'px_height']
feature2 = ["battery_power", "clock_speed", "int_memory", "m_dep", "mobile_wt", "n_cores", "pc" , "px_width", "ram", "sc_w", "talk_time","blue", "dual_sim", "three_g", "touch_screen", "wifi"]
X = validation[feature1]
y = validation["price_range"]
Xdata = data[feature1]
ydata = data["price_range"]
y_val = validation.iloc[:, -1].values
NB = GaussianNB()
NB.fit(Xdata,ydata)
ypred = NB.predict(X)
print(accuracy_score(y_val, ypred))


