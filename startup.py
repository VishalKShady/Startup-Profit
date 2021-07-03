import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

startup = pd.read_csv('50_Startups.csv')

le = LabelEncoder()
startup['State'] = le.fit_transform(startup['State'])

X = startup[['R&D Spend','Administration','Marketing Spend','State']].astype(int)
Y = startup['Profit'].astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.25,random_state=3)
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
y_pred = lr_model.predict(X_test)
print(y_pred)

#print('Intercept: ', lr_model.intercept_)
#print('Coefficients: ', lr_model.coef_)

#plt.scatter(y_test,y_pred)
#plt.plot(X_train, lr_model.predict(X_train))
#plt.show()