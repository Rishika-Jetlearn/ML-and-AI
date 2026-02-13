import pandas as pd

income=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\adult_income.csv",sep=", ")

income.info()
print(income["income"].value_counts())

X=income.drop(columns=["income","education"])
y=income["income"]

non_numeric=X.select_dtypes(include=["object"]).columns
print(non_numeric)
for i in non_numeric:

    print(income[i].value_counts())


income.loc[income["workclass"]=="?","workclass"]="unknown"
print(income["workclass"].value_counts())

income.loc[income["occupation"]=="?","occupation"]="Other-service"
print(income["occupation"].value_counts())

income.loc[income["native-country"]=="?","native-country"]="no-country"
print(income["native-country"].value_counts())
