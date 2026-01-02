import pandas as pd
from sklearn.linear_model import LinearRegression
house_data=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\HousingData.csv")
house_data=house_data[["RM","AGE","LSTAT","MEDV"]]
print(house_data.isna().sum())
house_data.info()
house_data.dropna(inplace=True)
house_data.info()

X=house_data[["RM","LSTAT"]]
y=house_data["MEDV"]
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(4)
transformed=pf.fit_transform(X)
print(X.head(3))
print(transformed[0:3])

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
X_train,X_test,y_train,y_test=train_test_split(transformed,y,train_size=0.8,random_state=32)
lr=LinearRegression()
lr.fit(X_train,y_train)
predicted_y=lr.predict(X_test)

rmse=root_mean_squared_error(y_test,predicted_y)
print(rmse)