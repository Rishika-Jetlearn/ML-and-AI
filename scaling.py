import numpy as np
array=np.array([100,142,107,507,109,382,220,900,213,655]).reshape(-1,1)
print(array)
print(np.mean(array))
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#changes the smallest to 0 and biggest to 1 and fits everything in between
mms=MinMaxScaler()
scaled_array=mms.fit_transform(array)
#changes the mean to 0 and fits everything in between(negative to positive)
ss=StandardScaler()
scaled=ss.fit_transform(array)

print(scaled_array)
print(scaled)