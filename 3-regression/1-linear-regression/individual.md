Solution to https://github.com/zipfian/linear-regression/blob/master/individual.md

  see linear_regression.py file









Matrix Vs Vectors
=====================================
1. Create a row vector with numpy (1 x some m)

    ```python
    In [2]: v = np.array([[1, 2, 3, 4]])
    ```

    or

    ```python
    In [3]: v = np.array([1, 2, 3, 4]).reshape((1, 4))
    ```

2. Create a column vector with numpy (some n x 1)

    ```python
    In [4]: v = np.array([[1], [2], [3], [4]])
    ```

    or

    ```python
    In [5]: v = np.array([1, 2, 3, 4]).reshape((4, 1))
    ```

3. Create a square 6 by 6 matrix

    ```python
    In [6]: np.matrix = np.ones((6, 6))
    ```

4. Create a random 2 x 3 matrix and a random 3 x 2 matrix

    ```python
    In [7]: np.random.randint(10, size=(2, 3))
    Out[7]:
    array([[9, 5, 2],
           [6, 0, 0]])

    In [8]: np.random.randint(10, size=(3, 2))
    Out[8]:
    array([[7, 7],
           [9, 2],
           [3, 1]])
    ```

5. Create an 6 x 6 identity matrix

    ```python
    In [9]: I = np.identity(6)
    ```

6. Create a matrix with any values and size and save it as `A`.

    ```python
    In [10]: A = np.array([[1, 2, 3], [4, 5, 6]])

    In [11]: A
    Out[11]:
    array([[1, 2, 3],
           [4, 5, 6]])
    ```

7. Get the number of rows and columns of the matrix `A`.

    ```python
    In [13]: A.shape
    Out[13]: (2, 3)
    ```

8. Create a transpose of the matrix `A`.

    ```python
    In [14]: A.transpose()
    Out[14]:
    array([[1, 4],
           [2, 5],
           [3, 6]])
    ```

9. Reshape the matrix `A` in to a 1 x n vector, where n is whatever it needs to be for the size of your matrix.

    ```python
    In [15]: A.reshape((1, 6))
    Out[15]: array([[1, 2, 3, 4, 5, 6]])
    ```


Scalar Operations
==============================
1. Create this numpy array (called `v`): `[2 3 5 8 9]`

    ```python
    In [16]: v = np.array([2, 3, 5, 8, 9])
    ```

2. Do a scalar addition by 0.5.

    ```python
    In [17]: v + 0.5
    Out[17]: array([ 2.5,  3.5,  5.5,  8.5,  9.5])
    ```

3. Do a scalar multiple by -2.

    ```python
    In [18]: v * -2
    Out[18]: array([ -4,  -6, -10, -16, -18])
    ```

4. Do a scalar divide by 0. What happens? Is that what you expected?

    ```python
    In [19]: v / 0
    /usr/local/bin/ipython:1: RuntimeWarning: divide by zero encountered in divide
      #!/usr/bin/python
    Out[19]: array([0, 0, 0, 0, 0])
    ```

    It gives 0 (though also gives a runtime warning).


5. Create a 1 by 5 vector `b` so that the following would get the same result as you did in #2: `v + b` (called broadcasting).

    ```python
    In [20]: b = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    In [21]: v + b
    Out[21]: array([ 2.5,  3.5,  5.5,  8.5,  9.5])
    ```

    or

    ```python
    In [22]: b = np.ones(5) * 0.5
    ```


Matrix Vector Multiplication
============================================
1. Create a random length 3 column vector, length 3 row vector, and 3 x 3 square matrix called `column_vector`, `row_vector`, and `rand_matrix`, respectively.

    ```python
    In [23]: column_vector = np.random.randint(10, size=(3, 1))

    In [24]: column_vector
    Out[24]:
    array([[7],
           [2],
           [3]])

    In [25]: row_vector = np.random.randint(10, size=(1, 3))

    In [26]: row_vector
    Out[26]: array([[8, 1, 0]])

    In [27]: rand_matrix = np.random.randint(10, size=(3, 3))

    In [28]: rand_matrix
    Out[28]:
    array([[9, 3, 9],
           [6, 8, 2],
           [0, 8, 7]])
    ```

2. Perform a vector vector multiply on `column_vector` and `row_vector`. This should output a `n x m` matrix where `n` is the number of rows in `row_vector` and `m` is the number of columns in `column_vector`. Say `column_vector` is a 2 x 1 and `row_vector` is a 1 x 3, output will be a 2 x 3 matrix.

    ```python
    In [29]: np.multiply(column_vector, row_vector)
    Out[29]:
    array([[56,  7,  0],
           [16,  2,  0],
           [24,  3,  0]])
    ```

3. For both `column_vector` and `row_vector`, matrix multiply by `rand_matrix`. One will have to go on the left and one will have to go on the right.

    ```python
    In [30]: np.dot(row_vector, rand_matrix)
    Out[30]: array([[78, 32, 74]])

    In [31]: np.dot(rand_matrix, column_vector)
    Out[31]:
    array([[96],
           [64],
           [37]])
    ```

4. Compute the dot product of `row_vector` and `column_vector`.

    ```python
    In [32]: np.dot(row_vector, column_vector)
    Out[32]: array([[58]])
    ```

    or

    ```python
    In [33]: row_vector.dot(column_vector)
    Out[33]: array([[58]])
    ```


Matrix Matrix Multiplication
======================================
1. If A is a 3 x 2 and B is a 4 x 3, can you matrix multiply them (AB)? If so, what is the shape? Can you matrix multiply them in the other direction (BA)? If so, what's the shape of that?

    You cannot matrix multiply AB, but you can matrix multiple BA. The result is a matrix of size 4 x 2.

2. Create a random 3 x 6 matrix as `rand_matrix`.

    ```python
    In [34]: rand_matrix = np.random.randint(10, size=(3, 6))

    In [35]: rand_matrix
    Out[35]:
    array([[5, 7, 2, 5, 5, 7],
           [8, 5, 0, 1, 5, 2],
           [3, 6, 2, 7, 7, 2]])
    ```

3. Matrix multiply `rand_matrix` and the transpose of `rand_matrix`.

    ```python
    In [36]: rand_matrix.dot(rand_matrix.T)
    Out[36]:
    array([[177, 119, 145],
           [119, 119, 100],
           [145, 100, 151]])
    ```

    or

    ```python
    In [37]: np.dot(rand_matrix, rand_matrix.T)
    Out[37]:
    array([[177, 119, 145],
           [119, 119, 100],
           [145, 100, 151]])
    ```

4. Reshape `rand_matrix` so that it can be multiplied by the original. Do the multiplication. The result should be a 3 x 3.

    ```python
    In [38]: np.dot(rand_matrix, rand_matrix.reshape((6, 3)))
    Out[38]:
    array([[145, 184,  93],
           [ 95, 130,  67],
           [103, 152,  80]])
    ```


Elementwise Matrix Operations
========================================
1. Create 2 random 6 x 2 matrices as `A` and `B`.

    ```python
    In [39]: A = np.random.randint(10, size=(6, 2))

    In [40]: B = np.random.randint(10, size=(6, 2))

    In [41]: A
    Out[41]:
    array([[2, 4],
           [6, 4],
           [3, 4],
           [7, 1],
           [8, 6],
           [2, 1]])

    In [42]: B
    Out[42]:
    array([[4, 6],
           [2, 2],
           [8, 6],
           [4, 2],
           [1, 5],
           [1, 5]])
    ```

2. Square `A` (this will be the same shape).

    ```python
    In [43]: A ** 2
    Out[43]:
    array([[ 4, 16],
           [36, 16],
           [ 9, 16],
           [49,  1],
           [64, 36],
           [ 4,  1]])
    ```

3. Add, subtract, multiply and divide `A` and `B` (This will be the same shape).

    ```python
    In [44]: A + B
    Out[44]:
    array([[ 6, 10],
           [ 8,  6],
           [11, 10],
           [11,  3],
           [ 9, 11],
           [ 3,  6]])

    In [45]: A - B
    Out[45]:
    array([[-2, -2],
           [ 4,  2],
           [-5, -2],
           [ 3, -1],
           [ 7,  1],
           [ 1, -4]])

    In [46]: A * B
    Out[46]:
    array([[ 8, 24],
           [12,  8],
           [24, 24],
           [28,  2],
           [ 8, 30],
           [ 2,  5]])

    In [47]: A / B
    Out[47]:
    array([[0, 0],
           [3, 2],
           [0, 0],
           [1, 0],
           [8, 1],
           [2, 0]])
    ```


Axis wise operations and operations across different dimensional matrices
================================
1. Create a 4 x 1 random matrix as `A`.

    ```python
    In [48]: A = np.random.randint(10, size=(4, 1))

    In [49]: A
    Out[49]:
    array([[4],
           [3],
           [9],
           [8]])
    ```

2. Create a 1 x 3 random matrix as `B`.

    ```python
    In [50]: B = np.random.randint(10, size=(1, 3))

    In [51]: B
    Out[51]: array([[6, 3, 5]])
    ```

3. Add the 2 (yes this is possible!). What is the shape of the result? What did it to?

    ```python
    In [52]: A + B
    Out[52]:
    array([[10,  7,  9],
           [ 9,  6,  8],
           [15, 12, 14],
           [14, 11, 13]])

    In [53]: _.shape
    Out[53]: (4, 3)
    ```

    It did every combination of adding an element form 'A' and an element from 'B'.

4. Create a random 2 x 3 matrix as `C`.

    ```python
    In [54]: C = np.random.randint(10, size=(2, 3))

    In [55]: C
    Out[55]:
    array([[8, 9, 3],
           [3, 4, 1]])
    ```

5. Calculate the sums, mean and standard deviation of the values in the matrix.

    ```python
    In [56]: C.sum()
    Out[56]: 28

    In [57]: C.mean()
    Out[57]: 4.666666666666667

    In [58]: C.std()
    Out[58]: 2.8674417556808756
    ```

6. Calculate the column wise sums, mean and standard deviation of the matrix.

```python
In [59]: C.sum(axis=0)
Out[59]: array([11, 13,  4])

In [60]: C.mean(axis=0)
Out[60]: array([ 5.5,  6.5,  2. ])

In [61]: C.std(axis=0)
Out[61]: array([ 2.5,  2.5,  1. ])
```

7. Calculate the row wise sums, mean and standard deviation of the matrix.

```python
In [62]: C.sum(axis=1)
Out[62]: array([20,  8])

In [63]: C.mean(axis=1)
Out[63]: array([ 6.66666667,  2.66666667])

In [64]: C.std(axis=1)
Out[64]: array([ 2.62466929,  1.24721913])
```


Rank
======================================
1. Create a random 5 x 3 matrix as `A`.

    ```python
    In [65]: A = np.random.randint(10, size=(5, 3))

    In [66]: A
    Out[66]:
    array([[8, 6, 7],
           [3, 5, 4],
           [3, 5, 1],
           [0, 6, 1],
           [4, 5, 1]])
    ```

2. Create a matrix `B` with a column added to the matrix to make it a 5 x 4 populating it with 2 * the first column.

    ```python
    In [67]: B = np.concatenate((A, np.reshape(A[:,1] * 2, (A.shape[0], 1))), axis=1)

    In [68]: B
    Out[68]:
    array([[ 8,  6,  7, 12],
           [ 3,  5,  4, 10],
           [ 3,  5,  1, 10],
           [ 0,  6,  1, 12],
           [ 4,  5,  1, 10]])
    ```

3. Calculate the rank of the matrix. This should be a number.

    ```python
    In [69]: np.linalg.matrix_rank(B)
    Out[69]: 3
    ```
