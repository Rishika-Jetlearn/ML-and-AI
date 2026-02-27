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



#pre processing
X["job"]=le.fit_transform(X["job"])

X["marital"]=le.fit_transform(X["marital"])

X["education"]=le.fit_transform(X["education"])

X["default"]=le.fit_transform(X["default"])

X["housing"]=le.fit_transform(X["housing"])

X["loan"]=le.fit_transform(X["loan"])

X["contact"]=le.fit_transform(X["contact"])

X["month"]=le.fit_transform(X["month"])

X["poutcome"]=le.fit_transform(X["poutcome"])

#scale
mms=MinMaxScaler()
X=mms.fit_transform(X)


#spliting data

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=76)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)

trained=classifier.fit(X_train,y_train)
tested=classifier.predict(X_test)

cr=classification_report(y_test,tested)
print(cr)

cm=confusion_matrix(y_test,tested)
print(cm)