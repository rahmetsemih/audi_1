import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#verilerin yüklenmesi
veriler = pd.read_csv("audi.csv")
veriler2=veriler.drop(['transmission','fuelType','model'],axis=1)
x = veriler2.iloc[:,0:5].values
y =veriler2.iloc[:,5:].values


#verileri train ve test olarak ayrılması
from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

predict = lr.predict(x_test)
