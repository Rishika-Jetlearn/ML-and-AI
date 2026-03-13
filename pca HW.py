import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

iris=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\Data-Science\iris.csv")
iris.info()



X=iris.drop(columns= ["species"])

mms=MinMaxScaler()
X=mms.fit_transform(X)

#now from df to array

y=iris["species"]


pca=PCA(n_components=3)
X=pca.fit_transform(X)

le=LabelEncoder()
encoded_y=le.fit_transform(y)


pf=PolynomialFeatures(2)
transformed=pf.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(transformed,encoded_y,train_size=0.8,random_state=32)
lr=LinearRegression()
lr.fit(X_train,y_train)
predicted_y=lr.predict(X_test)

#rmse
rmse=root_mean_squared_error(y_test,predicted_y)
print(rmse)

# wine=datasets.load_wine()
# print(wine.keys()