from numpy.random import beta as beta_dist
import numpy as np
import scipy.stats as st
from sklearn.linear_model import LinearRegression
import pandas as pd
import random
import itertools
import sqlite3 as sql

def run_sql_query(command, db):
    if not command:
        return []
    con = sql.connect(db)
    c = con.cursor()
    data = c.execute(command)
    result = [d for d in data]
    con.close()
    return result

# Probability

def roll_the_dice():
    '''
    INPUT: None
    OUTPUT: FLOAT

    Two unbiased dice are thrown once and the total score is observed. Use a
    simulation to find the estimated probability that the total score is even
    or greater than 7.
    '''
    total = 0
    num_repeats = 10000
    for i in xrange(num_repeats):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        score = die1 + die2
        if score % 2 == 0 or score > 7:
            total += 1
    return float(total) / num_repeats


# A/B Testing

def calculate_clickthrough_prob(clicks_A, views_A, clicks_B, views_B):
    '''
    INPUT: INT, INT, INT, INT
    OUTPUT: FLOAT

    Calculate and return an estimated probability that SiteA performs better
    (has a higher click-through rate) than SiteB.
    '''
    num_samples = 10000
    A_samples = beta_dist(clicks_A, views_A - clicks_A, num_samples)
    B_samples = beta_dist(clicks_B, views_B - clicks_B, num_samples)
    return np.mean(A_samples > B_samples)


# Statistics

def calculate_t_test(sample1, sample2, critical_value):
    '''
    INPUT: NUMPY ARRAY, NUMPY ARRAY
    OUTPUT: FLOAT, BOOLEAN

    You are asked to evaluate whether the two samples come from the same
    distribution.
    Return a tuple containing the p-value for the pair of samples and True or
    False depending if the p-value beats the critical value.
    '''
    blah, pvalue = st.ttest_ind(sample1, sample2)
    return pvalue, pvalue < critical_value


# Pandas and Numpy

def pandas_query(df):
    '''
    INPUT: DATAFRAME
    OUTPUT: DATAFRAME

    Given a DataFrame containing university data with these columns:
        name, address, Website, Type, Size

    Return the DataFrame containing the average size of university for each
    type ordered by size in ascending order.
    '''
    return df.groupby('Type').mean().sort('Size')
    # Alternative:
    # return df.groupby("Type")["Size"].mean().order()


def df_to_numpy(df, y_column):
    '''
    INPUT: DATAFRAME, STRING
    OUTPUT: 2 DIMENSIONAL NUMPY ARRAY, NUMPY ARRAY

    Make the column named y_column into a numpy array (y) and make the rest of
    the DataFrame into a 2 dimensional numpy array (X). Return (X, y).

    E.g.
                a  b  c
        df = 0  1  3  5
             1  2  4  6
        y_column = 'c'

        output: [[1, 3], [2, 4]],   [5, 6]
    '''
    y = df.pop(y_column)
    return df.values, y.values


def only_positive(arr):
    '''
    INPUT: 2 DIMENSIONAL NUMPY ARRAY
    OUTPUT: 2 DIMENSIONAL NUMPY ARRAY

    Return a numpy array containing only the rows from arr where all the values
    are positive.

    E.g.  [[1, -1, 2], [3, 4, 2], [-8, 4, -4]]  ->  [[3, 4, 2]]
    '''
    return arr[np.min(arr, 1) > 0]


def add_column(arr, col):
    '''
    INPUT: 2 DIMENSIONAL NUMPY ARRAY, NUMPY ARRAY
    OUTPUT: 2 DIMENSIONAL NUMPY ARRAY

    Return a numpy array containing arr with col added as a final column. You
    can assume that the number of rows in arr is the same as the length of col.

    E.g.  [[1, 2], [3, 4]], [5, 6]  ->  [[1, 2, 5], [3, 4, 6]]
    '''
    # by no means the only solution
    return np.hstack((arr, col.reshape(len(col), 1)))


def size_of_multiply(A, B):
    '''
    INPUT: 2 DIMENSIONAL NUMPY ARRAY, 2 DIMENSIONAL NUMPY ARRAY
    OUTPUT: TUPLE

    If matrices A (dimensions m x n) and B (dimensions p x q) can be
    multiplied, return the shape of the result of multiplying them. Use the
    shape function. Do not actually multiply the matrices, just return the
    shape.

    If A and B cannot be multiplied, return None.
    '''
    if A.shape[1] == B.shape[0]:
        return A.shape[0], B.shape[1]
    return None


# Linear Regression

def linear_regression(X_train, y_train, X_test, y_test):
    ''' 
    INPUT: 2 DIMENSIONAL NUMPY ARRAY, NUMPY ARRAY
    OUTPUT: TUPLE OF FLOATS, FLOAT

    Use the sklearn LinearRegression to find the best fit line for X_train and
    y_train. Calculate the R^2 value for X_test and y_test.

    Return a tuple of the coeffients and the R^2 value. Should be in this form:
    (12.3, 9.5), 0.567
    '''
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    return regr.coef_, regr.score(X_test, y_test)


# SQL

def sql_query():
    '''
    INPUT: None
    OUTPUT: STRING

    sqlite> PRAGMA table_info(universities);
    0,name,string,0,,0
    1,address,string,0,,0
    2,url,string,0,,0
    3,type,string,0,,0
    4,size,int,0,,0
    
    Return a SQL query that gives the average size of each type of university
    in ascending order.
    Columns should be: type, avg_size
    '''
    return '''SELECT type, AVG(size) AS avg_size FROM universities GROUP BY type ORDER BY avg_size;'''

