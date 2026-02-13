import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

bank=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\bank.csv",sep=";")

bank.info()


#X and y
X=bank.drop(columns=["y"])
y=bank["y"]

le=LabelEncoder()
y=le.fit_transform(y)

mms=MinMaxScaler()
X=mms.fit_transform(X)


#spliting data

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=76)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
trained=dtc.fit(X_train,y_train)
tested=dtc.predict(X_test)

cr=classification_report(y_test,tested)
print(cr)

cm=confusion_matrix(y_test,tested)
print(cm)