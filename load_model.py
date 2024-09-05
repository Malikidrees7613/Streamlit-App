import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model=joblib.load("K-Nearest Neighbors.pkl")

test_data= pd.read_csv("mobile_price_range_data.csv")

x_test= test_data.iloc[:,:-1]
y_test= test_data.iloc[:,-1]

y_pred = model.predict(x_test)

accuracy= accuracy_score(y_test,y_pred)

with open('accuracy.txt','w') as file:
    file.write(f'Accuracy: {accuracy}') 