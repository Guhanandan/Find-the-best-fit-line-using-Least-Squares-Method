import files
import numpy as np
import matplotlib.pyplot as plt

assign input

X=np.array([0,1,2,3,4,5,6,7,8,9])
Y=np.array([1,3,2,5,7,8,8,9,10,12])


 mean values of input

X_mean=np.mean(X)
print("X_mean =",X_mean)
Y_mean=np.mean(Y)
print("y_mean =",Y_mean)

num=0
denum=0

for i in range(len(X)):
  num+=(X[i]-X_mean)*(Y[i]-Y_mean)
  denum+=(X[i]-X_mean)**2

 find m
print("find m")
m=num/denum
print("m=",m)

#find b
print("find b")
b=(Y_mean)-(m*X_mean)
print("b =",b)

find Y_pred
print("find Y_pred")
Y_pred=m*X+b
print("Y_pred =",Y_pred)

plot graph

plt.scatter(X,Y,color='orange')
plt.plot(X,Y_pred,color='maroon')
print("Graph")
plt.show()
