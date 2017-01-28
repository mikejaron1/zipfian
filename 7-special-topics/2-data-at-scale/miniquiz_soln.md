Here's the solution to the prime generator:

```python
from math import sqrt

def is_prime(n):
    if n <= 1:
        return False
    for factor in xrange(2, int(sqrt(n)) + 1):
        if n % factor == 0:
            return False
    return True


def prime_generator():
    n = 1
    while True:
        if is_prime(n):
            yield(n)
        n += 1


if __name__ == '__main__':
    primes = prime_generator()
    print next(primes)
    print next(primes)
    print next(primes)
    print next(primes)
    print next(primes)
```

And the map reduce question:

```python
def acronym(phrase):
    letters = map(lambda x: x[0].upper(), phrase.split())
    return reduce(lambda x, y: x + y, letters)
```
