# Implementation of Univariate Linear Regression

## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1.Hardware – PCs

2.Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Get the independent variable X and dependent variable Y.

2.Calculate the mean of the X -values and the mean of the Y.

3.Find the slope m of the line of best fit using the formula.
<img width="231" alt="ml11" src="https://user-images.githubusercontent.com/100425381/200923714-81faee2d-1b8d-4cbf-b7f6-c1e4abbf0b79.png">

4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="ml12" src="https://user-images.githubusercontent.com/100425381/200923751-e2c6e4f3-a506-48db-adf6-6fb9cd40a8d5.png">


5. Use the slope m and the y -intercept to form the equation of the line. 

6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program :
~~~

/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: GUHANANDAN V
RegisterNumber:  212221220014
*/

import matplotlib.pyplot as plt
x=[5,6,3,2,6,7,1,2]
y=[2,3,6,5,8,3,5,8]
plt.scatter(x,y)
plt.show()

## Least Square Method

import numpy as np
import matplotlib.pyplot as plt
X=np.array([0,1,2,3,4,5,6,7,8,9])
Y=np.array([1,3,2,5,7,8,8,9,10,12])
 #mean 
X_mean=np.mean(X)
print(X_mean)
Y_mean=np.mean(y)
print(Y_mean)
num=0
denum=0
for i in range(len(x)):
  num+=(X[i]-X_mean)*(Y[i]-Y_mean)
  denum+=(x[i]-X_mean)**2
m=num/denum
b=Y_mean-m*X_mean
print(m,b)
Y_pred=m*X+b
print("y pred:",Y_pred)
plt.scatter(X,Y)
plt.plot(X,Y_pred,color='pink')
plt.show()
*/
~~~

## Output:

![ml13](https://user-images.githubusercontent.com/100425381/200923814-525e7a77-b991-4d0e-a024-10a91aafc27d.jpeg)

![ml14](https://user-images.githubusercontent.com/100425381/200923825-00265eab-077e-48b9-9df7-a413424f4bc8.jpeg)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
