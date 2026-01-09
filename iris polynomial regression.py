import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

iris=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\Data-Science\iris.csv")
iris.isna()
iris.info()
#no missing values
X=iris[["sepal_length","sepal_width","petal_length","petal_width"]]
y=iris["species"]

le=LabelEncoder()
encoded_y=le.fit_transform(y)
print(encoded_y)


pf=PolynomialFeatures(2)
transformed=pf.fit_transform(X)
print(X.head(3))
print(transformed[0:3])

X_train,X_test,y_train,y_test=train_test_split(transformed,encoded_y,train_size=0.8,random_state=32)
lr=LinearRegression()
lr.fit(X_train,y_train)
predicted_y=lr.predict(X_test)

rmse=root_mean_squared_error(y_test,predicted_y)
print(rmse)
