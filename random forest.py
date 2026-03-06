import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
income=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\adult_income.csv",sep=", ")

income.info()
print(income["income"].value_counts())

X=income.drop(columns=["income","education"])
y=income["income"]

non_numeric=X.select_dtypes(include=["object"]).columns
print(non_numeric)
for i in non_numeric:

    print(X[i].value_counts())


income.loc[X["workclass"]=="?","workclass"]="unknown"
print(X["workclass"].value_counts())

income.loc[X["occupation"]=="?","occupation"]="Other-service"
print(X["occupation"].value_counts())

income.loc[X["native-country"]=="?","native-country"]="no-country"
print(X["native-country"].value_counts())

le=LabelEncoder()
y=le.fit_transform(y)

for i in non_numeric:
    X[i]=le.fit_transform(X[i])

X.info()

#scale
mms=MinMaxScaler()
X=mms.fit_transform(X)


#spliting data

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=76)

#build classifier model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)

trained=classifier.fit(X_train,y_train)
tested=classifier.predict(X_test)

cr=classification_report(y_test,tested)
print(cr)

cm=confusion_matrix(y_test,tested)
print(cm)