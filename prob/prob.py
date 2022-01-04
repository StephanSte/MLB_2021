import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import matplotlib.pyplot as plt

data = pd.read_csv("housing.csv")
# gives the mean, median, standard deviation and percentiles of all the numerical values in your dataset.
print("mean, median, standard deviation and percentiles of all the numerical values in dataset")
print(data.describe())
# prints the column header and the data type stored in each column.
print("column header and the data type stored in each column")
print(data.info())
print("**************************************************************")
# check doubled entries
print("check doubled entries:")
print(data.duplicated(keep='last'))

print("**************************************************************")
print("reduce outliers")
max_height = data["Y house price of unit area"].quantile(0.95)
min_height = data["Y house price of unit area"].quantile(0.05)
print(max_height)
print(min_height)
print("data in range: ")
print(data[((data['Y house price of unit area'] < max_height) & (data['Y house price of unit area'] > min_height))])

# correct types
print("**************************************************************")
print("correct types: ")
print(data.dtypes)

# are there different colum parts like 1g and 0,0001 kg or smth
print("**************************************************************")
print("are there different colum parts like 1g and 0,0001 kg or smth: ")
test = is_string_dtype(data["Y house price of unit area"])
test2 = is_numeric_dtype(data["Y house price of unit area"])
print("Is it string or numeric?: ")
print(test)
print(test2)

# wurde der bearbeitet
print("**************************************************************")
print("wurde es bearbeitet?")
print("muss man nachfragen")

# fehlende werte
print("**************************************************************")
print("gibt es fehlende werte?")
print(data.isnull())
# groß genug?
print("**************************************************************")
print("Wie groß ist das und ist das genug?")
print(data.shape)
print("Obs genug ist kommt drauf an für was es verwendet wird")
print("**************************************************************")
print("Plots: ")
boxplot = data.boxplot(column=['Y house price of unit area', 'X2 house age'])
boxplot.plot()

hist = data.plot.bar(x='Y house price of unit area', y='X2 house age', rot=0)
hist.plot()

hist = data.plot.density(x='Y house price of unit area', y='X2 house age', rot=0)
hist.plot()

plt.show()
