import numpy as np
import matplotlib.pyplot as plt
x=np.arange(1,11)
y = np.array([23, 26, 27, 34, 38, 39, 45, 47, 48, 50])
plt.scatter(x,y)
plt.show()
#find line of best fit
#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#c = mean(y) – m * mean(x)
x_mean=np.mean(x)
y_mean=np.mean(y)
m=np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
c= y_mean-m*x_mean
print("slope: ",m)
print("intercept: ",c)
#prediction#
pred_y=m*x+c
print(pred_y)
plt.scatter(x,y)
plt.plot(x,pred_y)
plt.show()
#RMSE - Root Mean Squared Error sqrt( mean( (p – yi)^2 ))
rmse=np.sqrt(np.mean((pred_y-y)**2))
print(rmse)

#with ml
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
reshaped_x=x.reshape(-1,1)
lr.fit(reshaped_x,y)
print("slope =",lr.coef_)
print("intercept=",lr.intercept_)
new_predidicted_y=lr.predict(reshaped_x)
print(new_predidicted_y)

#get rmse from model
from sklearn.metrics import root_mean_squared_error
print(root_mean_squared_error(y,new_predidicted_y))

