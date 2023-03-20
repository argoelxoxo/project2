Author
:  Anjali Goel
# Gramfort’s version of Lowess, modified to work for multidimensional data!
https://gist.github.com/agramfort/850437

This is our given code for Gramfort's approach to Lowess!
```python
# Gramfort's approach
# only works for 1 dimensional inputs!
def  lowess_ag(x, y, f=2. / 3., iter=3):
	n = len(x)
	r = int(ceil(f * n))
	h = [np.sort(np.abs(x - x[i]))[r] for i in  range(n)]
	w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
	w = (1 - w ** 3) ** 3
	yest = np.zeros(n)
	delta = np.ones(n)
	for iteration in  range(iter):
		for i in  range(n):
			weights = delta * w[:, i]
			b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
			A = np.array([[np.sum(weights), np.sum(weights * x)],
			[np.sum(weights * x), np.sum(weights * x * x)]])
			beta = linalg.solve(A, b)
			yest[i] = beta[0] + beta[1] * x[i]			  
			residuals = y - yest
			s = np.median(np.abs(residuals))
			delta = np.clip(residuals / (6.0 * s), -1, 1)
			delta = (1 - delta ** 2) ** 2
	return yest
```
**So, what we want to do is modify this code to have it accommodate data that is not just one dimensional**
To do this, I took a look at our original code for Lowess:
```python
#our original version of lowess

class Lowess:
	def  __init__(self, kernel = Gaussian, tau=0.05):
		self.kernel = kernel
		self.tau = tau
	def  fit(self, x, y):
		kernel = self.kernel
		tau = self.tau
		self.xtrain_ = x
		self.yhat_ = y
	def  predict(self, x_new):
		check_is_fitted(self)
		x = self.xtrain_
		y = self.yhat_
		w = weights_matrix(x,x_new,self.kernel,self.tau)
		if np.isscalar(x_new):			lm.fit(np.diag(w).dot(x.reshape(-1,1)),np.diag(w).dot(y.reshape(-1,1)))
			yest = lm.predict([[x_new]])[0][0]
		elif  len(x.shape)==1:
			n = len(x_new)
			yest_test = np.zeros(n)
			#Looping through all x-points
			for i in  range(n):				lm.fit(np.diag(w[i,:]).dot(x.reshape(-1,1)),np.diag(w[i,:]).dot(y.reshape(-1,1)))
				yest_test[i] = lm.predict(x_new[i].reshape(-1,1))
		else:
			n = len(x_new)
			yest_test = np.zeros(n)
			#Looping through all x-points
			for i in  range(n):				lm.fit(np.diag(w[i,:]).dot(x),np.diag(w[i,:]).dot(y.reshape(-1,1)))
				yest_test[i] = lm.predict(x_new[i].reshape(1,-1))
		return yest_test
```
And then, I got to work to modify Alexander Gramfort's code to accommodate train and test sets with multidimensional features.
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from math import ceil
from scipy import linalg

lm = LinearRegression()
scale = StandardScaler()
qscale = QuantileTransformer()

def distance(u,v):
	if len(v.shape)==1:
		v = v.reshape(1,-1)
	d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))]) 
    return d
```
```python
def updatedlowess(x, y, newx,f=2/3, iter=3, intercept=True, qwerty = 6): 
	n = len(x) 
	r = int(ceil(f * n))
	yest = np.zeros(n)
	if len(y.shape)==1:
		y = y.reshape(-1,1)
	if len(x.shape)==1:
	    x = x.reshape(-1,1)
	if intercept:  
		x1 = np.column_stack([np.ones((len(x),1)),x])
	else: 
		x1 = x
		
	h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]
	w = np.clip(distance(x,x) / h, 0.0, 1.0) 
	w = (1 - w ** 3) ** 3
	delta = np.ones(n) 
	
	for iteration in range(iter): 	for i in range(n): 	
		W = delta * np.diag(w[:,i]) 
		b = np.transpose(x1).dot(W).dot(y)
		A = np.transpose(x1).dot(W).dot(x1)
		A = A + 0.0001*np.eye(x1.shape[1]) 
		beta = linalg.solve(A, b) 
		yest[i] = np.dot(x1[i],beta) 
	residuals = y - yest 
	s = np.median(np.abs(residuals)) 
	delta = np.clip(residuals / (qwerty * s), -1, 1)  
	delta = (1 - delta ** 3) ** 3 
	
	if x.shape[1]==1:
		f = interp1d(x.flatten(),yest,fill_value='extrapolate')
		output = f(newx)
	else:
		output = np.zeros(len(newx))
		
	for i in range(len(newx)):
		ind = np.argsort(np.sqrt(np.sum((x-newx[i])**2,axis=1)))[:r] 
		pca = PCA(n_components=3)
		x_pca = pca.fit_transform(x[ind])
		tri = Delaunay(x_pca,qhull_options='QJ')
		f = LinearNDInterpolator(tri,y[ind])
		output[i] = f(pca.transform(newx[i].reshape(1,-1))) 
		
	if sum(np.isnan(output))>0:
		g = NearestNDInterpolator(x,y.ravel()) 
		output[np.isnan(output)] = g(newx[np.isnan(output)])
	return output
```

Now, let's import the data we used before, split it up into test and train sets, and run our function on them!!
```python 
data = pd.read_csv('drive/MyDrive/cars.csv')
x = data.loc[:,'CYL':'WGT'].values
y = data['MPG'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
yhat = lowess_with_xnew(x_train,y_train,x_test,f=1/3,iter=5,intercept=True,qwerty = 2)
mse(y_test,yhat)
```
From this, we get a mean squared error of 17.265!!!


