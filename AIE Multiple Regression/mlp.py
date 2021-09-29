import pandas
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.datasets import make_regression


def forward_selection(data, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pandas.Series(index=remaining_features)
        for new_column in remaining_features:
            # ordinary least sqaures = ols
            #model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
            model = f_regression(data.iloc[:, :-1], data.iloc[:, -1])
            pvals = model[1]
            if new_column < 100:
                new_pval[new_column] = pvals[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features
#as

def backward_elimination(data, significance_level=0.05):
    while 2:
        p_vals = f_regression(data.iloc[:, :-1], data.iloc[:, -1])[1]
        if (p_vals <= significance_level).all() or data.shape[1] == 2:
            break
        bad_index = p_vals.argmax(axis=0)
        data = data.drop(data.columns[bad_index], axis=1)
    return data


n_features = 100
X, y = make_regression(n_samples=100, n_features=n_features, noise=25)
data = pandas.DataFrame(data=np.c_[X, y])

test = forward_selection(data)
print(test)

# exit(0)
# plt.scatter(X, y)
# plt.show()

regr = linear_model.LinearRegression()
regr.fit(X, y)

# slope for feature 0
k = regr.coef_[0]
# offset (height of line)
d = regr.intercept_

# plot samples and regression line
x = np.linspace(-5, 5, 100)
f_x = k * x + d
# line
plt.plot(x, f_x, '-r')
# points
# plt.scatter(X, y)
# plt.show()

# plot histogram
plt.hist(X, bins=10)
plt.show()

# calculate MSE and R2
n = len(X)
predictions = regr.predict(X)
# residual sum of squares form the slides
rss = np.sum((predictions - y) ** 2)
tss = np.sum((np.mean(y) - y) ** 2)
mse = rss / n
r_2 = 1 - (rss / tss)

print('MSE:', mse)
print('R2:', r_2)

f_value, p_value = f_regression(X, y)
print('P-Value: ', p_value)
