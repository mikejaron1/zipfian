## Fill each each function stub according to the docstring.
## Run the tests with this command: "make test"

import numpy as np
import pandas as pd

def max_lists(list1, list2):
    '''
    INPUT: list, list
    OUTPUT: list

    list1 and list2 have the same length. Return a list which contains the
    maximum element of each list for every index.
    '''
    return [max(list1[i], list2[i]) for i in xrange(len(list1))]

def get_diagonal(mat):
    '''
    INPUT: 2 dimensional list
    OUTPUT: list

    Given a matrix encoded as a 2 dimensional python list, return a list
    containing all the values in the diagonal starting at the index 0, 0.

    E.g.
    mat = [[1, 2], [3, 4], [5, 6]]
    | 1  2 |
    | 3  4 |
    | 5  6 |
    get_diagonal(mat) => [1, 4]

    You may assume that the matrix is nonempty.
    '''
    return [mat[i][i] for i in xrange(min(len(mat), len(mat[0])))]

def merge_dictionaries(d1, d2):
    '''
    INPUT: dictionary, dictionary
    OUTPUT: dictionary

    Return a new dictionary which contains all the keys from d1 and d2 with
    their associated values. If a key is in both dictionaries, the value should
    be the sum of the two values.
    '''
    d = d1.copy()
    for key, value in d2.iteritems():
        d[key] = d.get(key, 0) + value
    return d

def make_char_dict(filename):
    '''
    INPUT: string
    OUTPUT: dictionary (string => list)

    Given a file containing, you would like to create a dictionary with keys
    of single characters. The value is a list of all the line numbers which
    start with that letter. The first line should have line number 1.
    Characters which never are the first letter of a line do not need to be
    included in your dictionary.
    '''
    f = open(filename)
    num = 1
    d = {}
    for line in f:
        ch = line[0]
        if ch not in d:
                d[ch] = []
        d[ch].append(num)
        num += 1
    f.close()
    return d


### Pandas
# For each of these, you will be dealing with a DataFrame which contains median
# rental prices in the US by neighborhood. The DataFrame will have these
# columns:
# Neighborhood, City, State, med_2011, med_2014

def pandas_add_increase_column(df):
    '''
    INPUT: DataFrame
    OUTPUT: None

    Add a column to the DataFrame called 'Increase' which contains the 
    amount that the median rent increased by from 2011 to 2014.
    '''
    df['Increase'] = df['med_2014'] - df['med_2011']

def pandas_only_given_state(df, state):
    '''
    INPUT: DataFrame, string, string
    OUTPUT: DataFrame

    Return a new pandas DataFrame which contains the entries for the given
    state. Only include these columns:
        Neighborhood, City, med_2011, med_2014
    '''
    return df[df['State'] == state][['Neighborhood', 'City', 'med_2011', 'med_2014']]

def pandas_max_rent(df):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame

    Return a new pandas DataFrame which contains every city and the highest
    median rent from that city for 2011 and 2014.
    Your DataFrame should contain these columns:
        City, State, med_2011, med_2014

    '''
    return df[['City', 'State', 'med_2011', 'med_2014']].groupby(['City', 'State']).max()

    # Another solution:
    # return df.groupby(['City', 'State']).max().reset_index()[['City', 'State', 'med_2011', 'med_2014']]


### SQL
# For each of these, your python function should return a string that is the
# SQL statement which answers the question.
# For example:
#    return '''SELECT * FROM rent;'''
# You may want to run "sqlite3 data/housing.sql" in the command line to test
# out your queries if the test is failing.
#
# There are two tables in the database with these columns:
# (this is the same data that you dealt with in the pandas questions)
#     rent: Neighborhood, City, State, med_2011, med_2014
#     buy:  Neighborhood, City, State, med_2011, med_2014
# The values in the date columns are integers corresponding to the price on
# that date.

def sql_count_neighborhoods():
    '''
    INPUT: None
    OUTPUT: string

    Return a SQL query that gives the number of neighborhoods in each city
    according to the rent table. Keep in mind that city names are not always
    unique unless you include the state as well, so your result should have
    these columns: city, state, cnt
    '''
    return '''SELECT city, state, COUNT(1) AS cnt
              FROM rent
              GROUP BY city, state;'''

def sql_highest_rent_increase():
    '''
    INPUT: None
    OUTPUT: string

    Return a SQL query that gives the 5 San Francisco neighborhoods with the
    highest rent increase.
    '''
    return '''SELECT neighborhood
              FROM rent
              WHERE city='San Francisco'
              ORDER BY med_2014-med_2011 DESC LIMIT 5;'''

def sql_rent_and_buy():
    '''
    INPUT: None
    OUTPUT: string

    Return a SQL query that gives the rent price and buying price for 2014 for
    all the neighborhoods in San Francisco.
    Your result should have these columns:
        neighborhood, rent, buy
    '''
    return '''SELECT a.neighborhood, a.med_2014 AS rent, b.med_2014 AS buy
              FROM rent a
              JOIN buy b
              ON a.neighborhood=b.neighborhood AND a.city=b.city AND a.state=b.state
              WHERE a.city="San Francisco";'''

