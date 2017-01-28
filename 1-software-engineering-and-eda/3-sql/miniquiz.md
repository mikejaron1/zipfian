## 1. Create a Random Variable

Say you're giving a probability mass function in the form of a dictionary like this:

```python
{'A': 0.5, 'B': 0.1, 'C': 0.4}
```

I would like to get a random variable which is `A` 50% of the time, `B` 10% of the time and `C` 40% of the time. Write a function to do this. The only random function you can use is a uniform random distributino (use the `random` function from the `random` module).

In ipython, type `from random import random` to get the function and type `random?` to see what it does.

Now complete this function:

```python
def random_variable(pmf):
    '''
    INPUT: dictionary
    OUTPUT: one of the keys of the dictionary

    Return one of the keys of the dictionary according to the given probabilities. You may assume the probabilities sum to 1.
    '''

    num = random()
    total = 0
    for key, value in pmf.iteritems():
        total += value
        if num < total:
            return key
```

## 2. Python modules

Say you have a file called `code.py` with these contents:

```python
def print_hello(name):
    print "Hello, %s!" % name

print_hello("Jeff")
```

1. Before actually creating this file and running any code, determine what you think will happen if you run the following code in the command line.

    ```python
    from code import print_hello
    print_hello("Tammy")
    ```

    *This code will first print `Hello, Jeff!` and then print `Hello, Tammy`. THis is because all the code is run when it's imported.*

2. Create the `code.py` file and run the above code. Is this what you expected?

3. Modify the `code.py` so that you only get one "Hello" from running the above code. Make sure that if I run `python code.py` on the command line I will still get the same result as before!

    *You should put the code you don't want run in a main block like this:*

    ```python
    def print_hello(name):
    print "Hello, %s!" % name

    if __name__ == '__main__':
        print_hello("Jeff")
    ```

    *The code in the main block will be run when you run your code like this: `python code.py` but not when you import the file!*
