Extra Credit (> 2 dimensional data, yes this is used! Look into tensors) 
===========================
1. Create a random 3d tensor with 2 slices, 3 rows and 4 columns (you should create it just like you create a matrix except there's an extra dimension!)

    ```python
    In [5]: tensor = np.random.randint(10, size=(2, 3, 4))
    ```

2. Sum the 2 slices of the tensor.

    ```python
    In [6]: tensor[0] + tensor[1]
    Out[6]:
    array([[17,  8, 10,  5],
           [17,  6, 10, 14],
           [12,  6,  8, 10]])
    ```
