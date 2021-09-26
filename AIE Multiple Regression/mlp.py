import pandas
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.feature_selection import f_regression
from sklearn.datasets import load_boston


def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pandas.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


def backward_elimination(data, target, significance_level=0.05):
    features = data.columns.tolist()
    while len(features) > 0:
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if max_p_value >= significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features


boston = load_boston()
bos = pandas.DataFrame(boston.data, columns=boston.feature_names)
bos['Price'] = boston.target
X = bos.drop("Price", 1)  # feature matrix
y = bos['Price']  # target feature
bos.head()

#df = pandas.read_csv("cars.csv")

#X = df[['Weight', 'Volume']]
#y = df['CO2']

test = forward_selection(X, y)
print(test)
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

boston = load_boston()
bos = pandas.DataFrame(boston.data, columns=boston.feature_names)
bos['Price'] = boston.target
X = bos.drop("Price", 1)  # feature matrix
y = bos['Price']  # target feature
f_value, p_value = f_regression(X, y)
print('P-Value: ', p_value)
