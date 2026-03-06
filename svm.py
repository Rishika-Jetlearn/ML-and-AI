#getting datasets from sklearn
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

cancer_dict=datasets.load_breast_cancer()
print(cancer_dict.keys())
X=pd.DataFrame(cancer_dict.data,columns=cancer_dict.feature_names)
print(X)
y=pd.Series(cancer_dict.target)
print(y)

X.info()

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=20,random_state=67)

from sklearn.svm import SVC 
svc=SVC(kernel='linear')
svc.fit(Xtrain,ytrain)
predicted=svc.predict(Xtest)
print(predicted)

#evaluate the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)

trained=classifier.fit(Xtrain,ytrain)
tested=classifier.predict(Xtest)

cr=classification_report(ytest,tested)
print(cr)

cm=confusion_matrix(ytest,tested)
print(cm)