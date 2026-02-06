import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


cars=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\car.csv")

cars.info()
print(cars["boot_space"].value_counts())
print(cars["sales"].value_counts())
print(cars["persons"].value_counts())
print(cars["doors"].value_counts())
print(cars["maintenance"].value_counts())
print(cars["safety"].value_counts())
X=cars.drop(columns=["class"])
y=cars["class"]

le=LabelEncoder()
y=le.fit_transform(y)

X["boot_space"]=le.fit_transform(X["boot_space"])
X["sales"]=le.fit_transform(X["sales"])
X["maintenance"]=le.fit_transform(X["maintenance"])
X["safety"]=le.fit_transform(X["safety"])

"""X["sales"].astype(float)
cars.info()"""

X.loc[X["doors"]=="5more","doors"]=5
print(X["doors"].value_counts())
X["doors"]=X["doors"].astype(int)

X.loc[X["persons"]=="more","persons"]=6
print(X["persons"].value_counts())
X["persons"]=X["persons"].astype(int)


X.info()


#scaling
mms=MinMaxScaler()
X=mms.fit_transform(X)

#split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=48)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
trained=dtc.fit(X_train,y_train)
tested=dtc.predict(X_test)

cr=classification_report(y_test,tested)
print(cr)

cm=confusion_matrix(y_test,tested)
print(cm)