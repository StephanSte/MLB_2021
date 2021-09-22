import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


X, y = make_regression(n_samples=100, n_features=1, noise=21)
plt.scatter(X, y)
plt.show()

r = LinearRegression().fit(X, y)
# slope for feature 0
k = r.coef_[0]
# offset (height of line)
d = r.intercept_

# plot samples and regression line
x = np.linspace(-5, 5, 100)
f_x = k * x + d
# line
plt.plot(x, f_x, '-r')
# points
plt.scatter(X, y)
plt.show()

# plot histogram
plt.hist(X, bins=10)
plt.show()

# calculate MSE and R2
n = len(X)
predictions = r.predict(X)
# residual sum of squares form the slides
rss = np.sum((predictions - y) ** 2)
tss = np.sum((np.mean(y) - y) ** 2)
mse = rss / n
r_2 = 1 - (rss / tss)

print('MSE:', mse)
print('R2:', r_2)

X, y = make_regression(n_samples=100, n_features=1, noise=21)
f_value, p_value = f_regression(X, y)
print('P-Value: ', p_value)