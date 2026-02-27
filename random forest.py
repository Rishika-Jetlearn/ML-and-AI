import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
#build classifier model