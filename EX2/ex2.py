########################################################################################################################
# EX1
########################################################################################################################
import numpy as np
import pandas as pd

'''
load dataset "wine_exercise.csv" and try to import it correctly using pandas/numpy/...
'''
data = pd.read_csv('wine_exercise.csv')
data.shape()


'''
the dataset is based on the wine data with some more or less meaningful categorical variables
the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file
    - ...
'''

'''
(1) repair the dataset
    - consistent NA encodings. please note, na encodings might not be obvious at first ...
    - correct data types for all columns
    - correct categories (unique values) for object type columns
    - read all rows, including those with wrong/mixed decimal, separating characters
'''

'''
(2) find duplicates and exclude them
    - remove only the unnecessary rows
'''

'''
(3) find outliers and exclude them - write a function to plot histograms/densities etc. so you can explore a dataset quickly
    - just recode them to NA
    - proline (check the zero values), magnesium, total_phenols
    - for magnesium and total_phenols fit a normal and use p < 0.025 as a cutff value for idnetifying outliers
    - you should find 2 (magnesium) and  5 (total_phenols) outliers
'''

'''
(4) impute missing values using the KNNImputer
    - including the excluded outliers!
    - use only the original wine features as predictors! (no age, season, color, ...)
    - you can find the original wine features using load_wine()
    - never use the target for imputation!
'''

'''
(5) find the class distribution
    - use the groupby() method
'''

'''
(6) group magnesium by color and calculate statistics within groups
    - use the groupby() method
'''


########################################################################################################################
# Solution
########################################################################################################################


# set pandas options to make sure you see all info when printing dfs
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
