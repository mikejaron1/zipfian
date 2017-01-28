##3. Data Processing with Spark

**Setting up and load data**

```python
import pyspark as ps
import numpy as np

sc = ps.SparkContext('local[4]')

airline = sc.textFile('s3n://mortar-example-data/airline-data')
```

**Show first 2 entries**

```python
airline.take(2)

#[u'"YEAR","MONTH","UNIQUE_CARRIER","ORIGIN_AIRPORT_ID","DEST_AIRPORT_ID","DEP_DELAY","DEP_DELAY_NEW","ARR_DELAY","ARR_DELAY_NEW","CANCELLED",',
#u'2012,4,"AA",12478,12892,-4.00,0.00,-21.00,0.00,0.00,']
```

**Count all the entries**
```python
airline.count() #5113194
```

**Get rid of the quotes and the comma at the end**
```python
airline_no_quote = airline.map(lambda line: line.replace('\'', '').replace('\"', '').strip(','))
```


**Get the column names into a list**
```python
header_line = airline_no_quote.first()
header_lst = header_line.split(',')
print header_lst

#[u'YEAR', u'MONTH', u'UNIQUE_CARRIER', u'ORIGIN_AIRPORT_ID', u'DEST_AIRPORT_ID', u'DEP_DELAY', u'DEP_DELAY_NEW', u'ARR_DELAY', u'ARR_DELAY_NEW', u'CANCELLED']
```


**Get rid of the header line in the RDD**
```python
airline_no_header = airline_no_quote.filter(lambda row: row != header_line)
```

**Make row into dict**
```python
def flt2int(x):
    return int(float(x))

def make_row(row):
    row_lst = row.split(',')
    d = dict(zip(header_lst, row_lst))
    keys = ['DEST_AIRPORT_ID', 'ORIGIN_AIRPORT_ID', 'DEP_DELAY', 'ARR_DELAY']
    trimmed_d = {k: (d[k] if d[k] else 0.) for k in keys}
    trimmed_d['DEP_DELAY'] = flt2int(trimmed_d['DEP_DELAY'])
    trimmed_d['ARR_DELAY'] = flt2int(trimmed_d['ARR_DELAY'])
    trimmed_d['ARR_DELAY'] -= trimmed_d['DEP_DELAY']
    if int(float(d['CANCELLED'])) == 1:
        trimmed_d['DEP_DELAY'] += 300
    return trimmed_d
airline_cleaned = airline_no_header.map(lambda row: make_row(row))
destin_air = airline_cleaned.map(lambda d: (d['DEST_AIRPORT_ID'], d['ARR_DELAY']))
origin_air = airline_cleaned.map(lambda d: (d['ORIGIN_AIRPORT_ID'], d['DEP_DELAY']))


print destin_air.take(2)
print origin_air.take(2)

#[(u'12892', -17), (u'12892', -58)]
#[(u'12478', -4), (u'12478', -7)]
```

- **Departing airport that has least avgerage delay in minutes**
- **Departing airport that has most avgerage delay in minutes**
- **Arriving airport that has least avgerage delay in minutes**
- **Arriving airport that has most avgerage delay in minutes**

```python
destin_delays = destin_air.groupByKey()
origin_delays = origin_air.groupByKey()


destin_mean = destin_delays.mapValues(lambda delays: np.mean(delays.data))
origin_mean = origin_delays.mapValues(lambda delays: np.mean(delays.data))


destin_mean.persist()
origin_mean.persist()



best_destin_delays = destin_mean.sortBy(lambda tup: tup[1], ascending=True).take(10)
worst_destin_delays = origin_mean.sortBy(lambda tup: tup[1], ascending=False).take(10)

best_origin_delays = destin_mean.sortBy(lambda tup: tup[1], ascending=True).take(10)
worst_origin_delays = origin_mean.sortBy(lambda tup: tup[1], ascending=False).take(10)
```

**Output**
```python
worst_destin_delays

#[(u'10930', 60.735605170387778),
# (u'13388', 60.03344481605351),
# (u'13964', 52.389344262295083),
# (u'13424', 49.313011828935394),
# (u'10157', 45.667342114150628),
# (u'14487', 39.081979891724671),
# (u'11002', 34.005529953917048),
# (u'13541', 33.845454545454544),
# (u'10170', 33.217857142857142),
# (u'10165', 31.931818181818183)]
```

