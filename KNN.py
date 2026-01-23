import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
iris=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\iris.csv")
X=iris.drop(columns=["species"])#drop target keep the rest
print(X)
y=iris["species"]

le=LabelEncoder()
y=le.fit_transform(y)

#scaling
mms=MinMaxScaler()
X=mms.fit_transform(X)


#HW: 
#1. split data into train and test
#2. build model(using practice doc)
#3. fit,predict,confusion matrix ect.

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
print(max)
k=f1_scores.index(max)+1
print(k)

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
sns.heatmap(cm,annot=True)
plt.show()


