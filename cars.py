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

#finding best k value 
from sklearn.metrics import f1_score
hightest_k=int(np.sqrt(X_train.shape[0]))
f1_scores=[]

for i in range(1,hightest_k):
    cf=KNeighborsClassifier(n_neighbors=i)
    cf.fit(X_train,y_train)
    test=cf.predict(X_test)
    f1_scores.append(f1_score(y_test,test,average="macro"))

max=max(f1_scores)
k=f1_scores.index(max)+1

classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(X_train,y_train)
trained=classifier.predict(X_train)
tested=classifier.predict(X_test)

print(trained)
print(tested)

#confusion matric +classification report
cr=classification_report(y_test,tested)
cm=confusion_matrix(y_test,tested)
print(cm)
print(cr)

#creating heatmap
sns.heatmap(cm,annot=True,fmt="d")

plt.show()
