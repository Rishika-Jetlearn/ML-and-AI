import pandas as pd
from sklearn.linear_model import LinearRegression
house_data=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\HousingData.csv")
house_data=house_data[["RM","AGE","LSTAT","MEDV"]]
print(house_data.isna().sum())
house_data.info()
house_data.dropna(inplace=True)
house_data.info()

X=house_data[["RM","AGE","LSTAT"]]
y=house_data["MEDV"]
#splitting data into training ans testing set
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=56)

lr=LinearRegression()
lr.fit(Xtrain,ytrain)
predicition_training=lr.predict(Xtrain)
predicition_testing=lr.predict(Xtest)
#rmse
print(root_mean_squared_error(ytrain,predicition_training))
print(root_mean_squared_error(ytest,predicition_testing))