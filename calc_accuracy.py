import pandas as pd
import sys

pred = pd.read_csv(sys.argv[1]) 
true = pd.read_csv("Data/" + sys.argv[2])

true_positive = 0
true_negative = 0
for i in range(len(pred)):
    if pred['price_range'][i] == true['price_range'][pred['id'][i]]:
        true_positive += 1
    else:
        true_negative += 1

print("accuracy: ",true_positive / (true_positive + true_negative))
