import sys


def fizzbuzz(num, fizz=3, buzz=5):
    '''
    INPUT: int, int, int
    OUTPUT: string

    Return "Fizz" if num is divisible by fizz,
           "Buzz" if num is divisible by buzz,
           "FizzBuzz" if num is dibisible by both fizz and buzz, and
           "" otherwise
    '''
    if num % fizz == 0:
        if num % buzz == 0:
            return "FizzBuzz"
        else:
            return "Fizz"
    else:
        return "Buzz"
    return ""


if __name__ == '__main__':
    print fizzbuzz(int(sys.argv[1]))
