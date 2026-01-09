import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

titanic_csv=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\titanic.csv")
titanic_csv.isna()
titanic_csv.info()

X=titanic_csv[["Pclass","Sex","Age","Siblings/Spouses Aboard","Parents/Children Aboard","Fare"]]
y=titanic_csv["Survived"]
le=LabelEncoder()
X["Sex"]=le.fit_transform(X["Sex"])

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=48)
lr=LogisticRegression()
lr.fit(X_train,y_train)
trained=lr.predict(X_train)
tested=lr.predict(X_test)
print(trained)
print(tested)

#create a confusion matric
from sklearn.metrics import confusion_matrix,classification_report
cr=classification_report(y_test,tested)
cm=confusion_matrix(y_test,tested)
print(cm)
print(cr)