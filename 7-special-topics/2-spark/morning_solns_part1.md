
##1. RDD and Spark Basics

**Import libraries**

```python
import pyspark as ps
import json
from operator import add
```

**Define SparkContext**

```python
sc = ps.SparkContext('local[4]')
```

**Turn a Python list into an RDD**

```python
lst_rdd = sc.parallelize([1, 2, 3])
```

**Read in the textFile as an RDD**
```python
file_rdd = sc.textFile('data/toy_data.txt')
```

**Views the first entries**
```python
print file_rdd.first()
print file_rdd.take(2)

#{"Jane": "2"}
#[u'{"Jane": "2"}', u'{"Jane": "1"}']
```


**Get all entries**

```python
print file_rdd.collect()
print lst_rdd.collect()

#[u'{"Jane": "2"}', u'{"Jane": "1"}', u'{"Pete": "20"}', u'{"Tyler": "3"}', u'{"Duncan": "4"}', u'{"Yuki": "5"}', u'{"Duncan": "6"}', u'{"Duncan": "4"}', u'{"Duncan": "5"}']
#[1, 2, 3]
```


##2. Intro to Functional Programming

**Load each line as a dictionary and then cast as (key, value)**

```python
tup_rdd = file_rdd.map(lambda line: json.loads(line))\
                  .map(lambda d: (d.keys()[0], int(d.values()[0])))
tup_rdd.first()
#(u'Jane', 2)
```

**Entries with more than 5 cookies**

```python
print tup_rdd.filter(lambda x: x[1] > 5).collect()
#[(u'Pete', 20), (u'Duncan', 6)]
```

**For each name, return the entry with the max number of cookies**

```python
print tup_rdd.groupByKey().mapValues(lambda tup: max(tup.data)).collect()
#[(u'Jane', 2), (u'Pete', 20), (u'Yuki', 5), (u'Tyler', 3), (u'Duncan', 6)]
```

**Total revenue from people buying cookies**

```python
print tup_rdd.values().reduce(add)
#50
```
