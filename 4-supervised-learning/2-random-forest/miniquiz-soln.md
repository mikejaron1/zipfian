# Miniquiz

You have a stream of items of large and unknown length that we can only iterate over once. Assume that the stream is large enough that it doesn't fit into main memory. For example, a list of search queries in Google or interactions on Facebook.

1. Given a data stream of unknown size `n`, write a function that picks an entry uniformly at random. This is, each entry has a `1/n` chance of being chosen. 

```python
import random

def reservoirSample(stream):
   for k,x in enumerate(stream, start=1):
	  if random.random() < 1.0 / k:
		 chosen = x

   return chosen
```

2. Extend the algorithm to pick `k` samples from this stream such that each item is equally likely to be selected.

```python
samples = []
def reservoirSample(stream, k):
	for i, x in enumerate(stream):
		# Generate the reservoir
		if index <= k:
			samples.append(x)
		else:                  
			# Randomly replace elements in the reservoir
			# with a decreasing probability.             
			# Choose an integer between 0 and index               
			replace = random.randint(0, i-1)               
			if replace < k:                       
				sample[replace] = x

print samples
```

## Discussion

This is one of many techniques used to solve a problem called _reservoir sampling_. We often encounter data sets that we’d like to sample elements from at random. But with the advent of big data, the lists involved are so large that we either can’t fit it all at once into memory or we don’t even know how big it is because the data is in the form of a stream (e.g., the number of atomic price movements in the stock market in a year). Reservoir sampling is the problem of sampling from such streams, and the technique above is one way to achieve it.

In words, the above algorithm holds one element from the stream at a time, and when it inspects the `k`-th element (indexing k from 1), it flips a coin of bias `1/k` to decide whether to keep its currently held element or to drop it in favor of the new one.

We can prove quite easily by induction that this works. Indeed, let `n` be the (unknown) size of the list, and suppose `n=1`. In this case there is only one element to choose from, and so the probability of picking it is `1`. The case of `n=2` is similar, and more illustrative. Now suppose the algorithm works for `n` and suppose we increase the size of the list by 1 adding some new element `y` to the end of the list. For any given `x` among the first `n` elements, the probability we’re holding `x` when we  inspect `y` is `1/n` by induction. Now we flip a coin which lands heads with probability `1/(n+1)`, and if it lands heads we take `y` and otherwise we keep `x`. The probability we get `y` is exactly `1/(n+1)`, as desired, and the probability we get `x` is `(1/n) * (n/n+1) = (1/n+1)`. Since `x` was arbitrary, this means that after the last step of the algorithm each entry is held with probability `1/(n+1)`.

### References

* [Algorithms Every Data Scientist Should Know: Reservoir Sampling](http://blog.cloudera.com/blog/2013/04/hadoop-stratified-randosampling-algorithm/)
* [Reservoir Sampling](http://gregable.com/2007/10/reservoir-sampling.html)
* [Weighted Reservoir Sampling](http://arxiv.org/pdf/1012.0256.pdf)

