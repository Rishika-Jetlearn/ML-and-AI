from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lr=LinearRegression()
salary_df=pd.read_csv(r"C:\Users\Manisha\Desktop\Jetlearn-python-rishika\ML-and-AI\Salary.csv")
x=salary_df["YearsExperience"]
y=salary_df["Salary"]

plt.scatter(x,y)
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.title("Salary for experience")
plt.show()

#make x 2D

two_dimentional_x=np.array(x).reshape(-1,1)
#make it learn
lr.fit(two_dimentional_x,y)
#get slope and intercept
print("slope =",lr.coef_)
print("intercept=",lr.intercept_)

#predict y
new_y=lr.predict(two_dimentional_x)
print(new_y)

#rmse
from sklearn.metrics import root_mean_squared_error
print(root_mean_squared_error(y,new_y))