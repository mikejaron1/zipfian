##Spark Afternoon Solutions

###Training Model

**Import libraries**

```python
import string
import json 
import pickle as pkl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pyspark as ps
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
```

**Define function to tokenize text**

```python
# Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them
def tokenize(text):
    PUNCTUATION = set(string.punctuation)
    STOPWORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]
```

- **Setting local spark context**
- **Read in text file**

```python
conf = ps.SparkConf()
conf.setMaster('local[4]')
sc = ps.SparkContext(conf=conf)
data_raw = sc.textFile('s3n://newsgroup/news.txt')
```

**Parse JSON entries in dataset**
```python
data = data_raw.map(lambda line: json.loads(line))
```

**Label number map to label name**
```python
label_to_name = data.map(lambda line: (line['label'], line['label_name'])).distinct().collect()
```

**Extract relevant fields in dataset -- category label and text content**
```python
data_pared = data.map(lambda line: (line['label'], line['text']))
```

**Prepare text for analysis using our tokenize function to clean it up**
```python
data_cleaned = data_pared.map(lambda (label, text): (label, tokenize(text)))
```

**Hashing term frequency vectorizer with 50k features**
```python
htf = HashingTF(50000)
```

**Create an RDD of LabeledPoints using category labels as labels and tokenized, hashed text as feature vectors**
```python
data_hashed = data_cleaned.map(lambda (label, text): LabeledPoint(label, htf.transform(text)))
```

**Ask Spark to persist the RDD so it won't have to be re-created later**
```python
data_hashed.persist()
```

**Split data 70/30 into training and test data sets**
```python
train_hashed, test_hashed = data_hashed.randomSplit([0.7, 0.3])
train_hashed.count() #9107
```

**Train a Naive Bayes model on the training data**
```python
model = NaiveBayes.train(train_hashed)
```

**Compare predicted labels to actual labels**
```python
prediction_and_labels = test_hashed.map(lambda point: (model.predict(point.features), point.label))
```

**Filter to only correct predictions**
```python
correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
```

**Calculate and print accuracy rate**
```python
accuracy = correct.count() / float(test_hashed.count())
print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"
# Classifier correctly predicted category 84.5772442589 percent of the time
```

**Dumping model into files to use later**

```python
pkl.dump(model, open('naive_bayes.pkl', 'wb'))
pkl.dump(dict(label_to_name), open('label_to_name.pkl', 'wb'))
```

###Predict new data

**Reload trained model and load new data**
```python
naive_bayes = pkl.load(open('naive_bayes.pkl', 'rb'))
new_coming_data = pkl.load(open('news_test.pkl'))
new_data_rdd = sc.parallelize(new_coming_data)
```


**Preprocess new data**
```python
new_htf = HashingTF(50000)
tokenized_rdd = new_data_rdd.map(lambda text: tokenize(text))
features = tokenized_rdd.map(lambda text: new_htf.transform(text))
```

**Predict**
```python
predictions = features.map(lambda feature: int(naive_bayes.predict(feature)))
predictions_lst = predictions.collect()
```
