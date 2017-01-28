# How to build a text classifier web app.  
Today we are going to train a model, build a web interface that allows people to submit data, send that data to our model, make a prediction on it, then return the results.

##### This is how your app directory should look.  
```
MyProject/
|-- my_app/
|   |-- app.py
|   |-- build_model.py
|   |-- data
|   |   |-- my_data.csv
|   |   |-- my_model.pkl
|   |   |-- my_vectorizer.pkl
```
<br>
## Step 1: Build your model
Step 1 should take about *30min–60min*  
`MyProject/my_app/build_model.py`
1.  Build ANY text classifier model.  
2.  Pickle and export your trained model and vectorizer into your data folder.  
3. *See below if you want a step by step guide for this*

<br>
## Step 2:  Build your site
Step 2 should take anywhere from 30min–120min  
1.  Create an app.py file in your my_app folder  
    *  **MyProject/my_app/app.py**
1.  Build a simple web homepage using flask.
2.  Once you have setup a working homepage...
3.  Build a submission_page that has an html form for the user to submit new text data.
4.  Build a predict_page that processes the user submitted form data, and returns the result of your prediction.  


---

Run your app.py script and send us your apps address because you are done.  

<br>
<br>
<br>

###Step 1:  Step by step

1. Import pickle and use it to load the articles.pkl file.
https://wiki.python.org/moin/UsingPickle

2. Set your text data column to `X`.

3. Set your label data column to `y`.

2. Initialize a multinomial naive bayes classifier.  

3. Initialize a TFIDF vectorizer.

4. With your TFIDF vectorizer, fit and transform your `X` text data. Name the output `vectorized_X`
.
5.  Initialize your MultinomialNB model
```clf = MultinomialNB()```

6.  Fit your model with the `transformed_X` data, and the `y` labels.  

7.  Export your fitted model using pickle.

8.  Export your fitted vectorizer using pickle.

9.  Take a mini break.
---
<br>
<br>
<br>
