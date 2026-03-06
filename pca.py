import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

diabetes=datasets.load_diabetes()
print(diabetes.keys())

X=pd.DataFrame(data=diabetes.data,columns=diabetes.feature_names)

X.info()
print(X.head(3))

mms=MinMaxScaler()
X=mms.fit_transform(X)
#now from df to array
print(X[0:3])

y=pd.Series(data=diabetes.target)
print(y)
from sklearn.decomposition import PCA

pca=PCA(n_components=3)
X=pca.fit_transform(X)
print(X[0:3])

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=20,random_state=57)

#model
svr=SVR(kernel="linear")
svr.fit(Xtrain,ytrain)
predicted=svr.predict(Xtest)

#evaluate 
print(root_mean_squared_error(ytest,predicted))
