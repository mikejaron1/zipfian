### Python

def count_characters(string):
    '''
    INPUT: STRING
    OUTPUT: DICT (STRING => INT)

    Return a dictionary which contains a count of the number of times each
    character appears in the string.
    Characters which would have a count of 0 should not need to be included in
    your dictionary.
    '''
    d = {}
    for char in string:
        d[char] = d.get(char, 0) + 1
    return d

def invert_dictionary(d):
    '''
    INPUT: DICT (STRING => INT)
    OUTPUT: DICT (INT => SET OF STRINGS)

    Given a dictionary d, return a new dictionary with d's values as keys and
    the value for a given key being the set of d's keys which have the same
    value.
    e.g. {'a': 2, 'b': 4, 'c': 2} => {2: {'a', 'c'}, 4: {'b'}}
    '''
    result = {}
    for key, value in d.iteritems():
        if value not in result:
            result[value] = set()
        result[value].add(key)
    return result

def word_count(filename):
    '''
    INPUT: STRING
    OUTPUT: (INT, INT, INT)

    filename refers to a text file.
    Return a tuple containing these stats for the file in this order:
      1. number of lines
      2. number of words (broken by whitespace)
      3. number of characters
    '''
    f = open(filename)
    l = 0
    w = 0
    c = 0
    for line in f:
        l += 1
        w += len(line.split())
        c += len(line)
    return l, w, c

def matrix_multiplication(A, B):
    '''
    INPUT: LIST OF LIST OF INTEGERS, LIST OF LIST OF INTEGERS
    OUTPUT: LIST OF LIST of INTEGERS

    A and B are matrices with integer values, encoded as lists of lists:
    e.g. A = [[2, 3, 4], [6, 4, 2], [-1, 2, 0]] corresponds to the matrix:
    | 2  3  4 |
    | 6  4  2 |
    |-1  2  0 |
    Return the matrix which is the product of matrix A and matrix B.
    You may assume that A and B are square matrices of the same size.
    '''
    size = len(A)
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            sum = 0
            for k in range(size):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    return result


### Probability

def cookie_jar(a, b):
    '''
    INPUT: FLOAT, FLOAT
    OUTPUT: FLOAT

    There are two jars of cookies with chocolate and peanut butter cookies.
    a: fraction of Jar A which is chocolate
    b: fraction of Jar B which is chocolate
    A jar is chosen at random and a cookie is drawn.
    The cookie is chocolate.
    Return the probability that the cookie came from Jar A.
    '''
    return a / (a + b)


### NumPy and Pandas

def array_work(rows, cols, scalar, matrixA):
    '''
    INPUT: INT, INT, INT, NUMPY ARRAY
    OUTPUT: NUMPY ARRAY

    Create matrix of size (rows, cols) with the elements initialized to the
    scalar value. Right multiply that matrix with the passed matrixA (i.e. AB,
    not BA). 
    Return the result of the multiplication.
    You should be able to accomplish this in a single line.

    Ex: array_work(2, 3, 5, [[3, 4], [5, 6], [7, 8]])
           [[3, 4],      [[5, 5, 5],
            [5, 6],   *   [5, 5, 5]]
            [7, 8]]
    '''
    return matrixA.dot(np.ones((rows, cols)) * scalar)


def make_series(start, length, index):
    '''
    INPUT: INT, INT, LIST

    Create a pandas Series of length "length"; its elements should be
    sequential integers starting from "start". 
    The series' index should be "index". 

    Ex: 
    In [1]: make_series(5, 3, ['a', 'b', 'c'])
    Out[1]: 
    a    5
    b    5
    c    7
    dtype: int64
    '''
    return pd.Series(np.arange(length) + start, index=index)


def create_data_frame(csv_file):
    '''
    INPUT: FILE OBJ
    OUTPUT: DATAFRAME

    Return a pandas DataFrame constructed from the passed csv file.
    '''
    return pd.DataFrame.from_csv(csv_file)


def data_frame_work(df, colA, colB, colC):
    '''
    INPUT: DATAFRAME, STR, STR, STR
    OUTPUT: None
    
    Insert a column (colC) into the dataframe that is the sum of colA and colB.
    '''
    df[colC] = df[colA] + df[colB]


def reverse_index(arr, finRow, finCol):
    '''
    INPUT: NUMPY ARRAY, INT, INT
    OUTPUT: NUMPY ARRAY

    Reverse the row order of "arr" (i.e. so the top row is on the bottom)
    and return the sub-matrix from coordinate [0, 0] to [finRow, finCol],
    exclusive.

    Ex:
    In [1]: arr = np.array([[ -4,  -3,  11],
                            [ 14,   2, -11],
                            [-17,  10,   3]])
    In [2]: reverse_index(arr, 2, 2)
    Out[2]: 
    array([[-17,  10],
           [ 14,   2]])
    '''
    return arr[::-1][:finRow, :finCol]

def boolean_indexing(arr, minimum):
    '''
    INPUT: NUMPY ARRAY, INT
    OUTPUT: NUMPY ARRAY

    Returns an array with all the elements of "arr" greater than
    or equal to "minimum"

    Ex:
    In [1]: boolean_indexing([[3, 4, 5], [6, 7, 8]], 7)
    Out[1]: array([7, 8])
    '''
    return arr[arr >= minimum]


## SQL
def markets_per_state():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL statement which gives the states and the number of markets
    for each state which take WIC or WICcash.
    '''

    return '''SELECT State, COUNT(1)
              FROM farmersmarkets
              WHERE WIC='Y' OR WICcash='Y'
              GROUP BY State;'''


def markets_taking_wic():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL statement which gives the percent of markets which take WIC
    or WICcash.
    The WIC and WICcash columns contain either 'Y' or 'N'
    '''

    return '''SELECT
                  (SELECT COUNT(1) FROM farmersmarkets WHERE WIC='Y' OR WICcash='Y') /
                  (SELECT CAST(COUNT(1) AS REAL) FROM farmersmarkets);'''

def state_population_gain():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL statement which gives the 10 states with the highest
    population gain from 2000 to 2010.
    '''

    return '''SELECT state FROM statepopulations ORDER BY (pop2010-pop2000) DESC LIMIT 10;'''

def markets_populations():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL statement which gives a table containing each market name,
    the state it's in and the state population from 2010
    Sort by MarketName
    '''

    return '''SELECT MarketName, farmersmarkets.State, pop2010
              FROM farmersmarkets
              JOIN statepopulations ON farmersmarkets.State=statepopulations.state
              ORDER BY MarketName;'''

def market_density_per_state():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL statement which gives a table containing each state, number
    of people per farmers market (use the population number from 2010).
    If a state does not appear in the farmersmarket table, it should still
    appear in your result with a count of 0.
    '''

    return '''SELECT p.state, IFNULL(p.pop2010 / m.cnt, 0)
              FROM statepopulations p
              LEFT OUTER JOIN (SELECT state, COUNT(1) AS cnt FROM farmersmarkets GROUP BY state) m
              ON p.state=m.state;'''
