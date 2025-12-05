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
