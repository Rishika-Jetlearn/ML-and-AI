import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


admisson_df=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\admission.csv")
admisson_df.isna()
admisson_df.info()
#no missing

X=admisson_df[["gre","gpa","rank"]]
y=admisson_df["admit"]

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=48)

lr=LogisticRegression()
lr.fit(X_train,y_train)

trained=lr.predict(X_train)
tested=lr.predict(X_test)

print(trained)
print(tested)

#confusion matric
cr=classification_report(y_test,tested)
cm=confusion_matrix(y_test,tested)
print(cm)
print(cr)

