#HW:Perform multi-variable linear regression on the Iris dataset.
#features=petal width/length,sepal width/length       target=species
#predict the species using feature/target

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

iris=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\Data-Science\iris.csv")
iris.isna()
iris.info()
#no missing values
X=iris[["sepal_length","sepal_width","petal_length","petal_width"]]
y=iris["species"]

le=LabelEncoder()
encoded=le.fit_transform(y)
print(encoded)
#splitting data into training and testing set
X_train,X_test,y_train,y_test=train_test_split(X,encoded,train_size=0.8,random_state=34)
lr=LinearRegression()
lr.fit(X_train,y_train)
predicition_training=lr.predict(X_train)
predicition_testing=lr.predict(X_test)

