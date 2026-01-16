import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

iris=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\iris.csv")
X=iris.drop(columns=["species"])#drop target keep the rest
print(X)
y=iris["species"]

le=LabelEncoder()
y=le.fit_transform(y)

#scaling
mms=MinMaxScaler()
X=mms.fit_transform(X)
