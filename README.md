# Implementation of Univariate Linear Regression

## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Get the independent variable X and dependent variable Y.

2.Calculate the mean of the X -values and the mean of the Y -values.

3.Find the slope m of the line of best fit using the formula.

<img width="231" alt="ml11" src="https://user-images.githubusercontent.com/100425381/198864331-6fd4da21-cda7-4dc2-ad59-72ee4fed0c29.png">

4. Compute the y -intercept of the line by using the formula

<img width="148" alt="ml12" src="https://user-images.githubusercontent.com/100425381/198864336-2722caac-3db0-4473-bee4-e192e8230849.png">


5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
~~~

/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: Guhanandan V
RegisterNumber:  212221220014
*/
import numpy as np
import matplotlib.pyplot as plt
#Preprocessing Input data
X=np.array(eval(input()))
Y=np.array(eval(input()))
#Mean
X_mean=np.mean(X)
Y_mean=np.mean(Y)
num=0 #for slope
denom=0 #for slope
#to find sum of (xi-x') & (yi-y') & (xi-x')^2
for i in range(len(X)):
  num+=(X[i]-X_mean)*(Y[i]-Y_mean)
  denom+=(X[i]-X_mean)**2
#calculate slope
m=num/denom

#calculate intercept
b=Y_mean-m*X_mean

print(m,b)

#line equation
y_predicted=m*X+b
print(y_predicted)

#to plot graph
plt.scatter(X,Y)
plt.plot(X,y_predicted,color='red')
plt.show()
~~~
## Output:
best fit line
![ml13](https://user-images.githubusercontent.com/100425381/198864347-42c40687-4790-4f7d-ad47-0874dd95ee1d.png)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
