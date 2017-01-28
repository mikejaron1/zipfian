
# Step 1: Build and save your model.  
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




# STEP 2: Setting up a flask app.
Before we do anything, we must pip install flask.
`pip install flask`

Here is a starter file with the most simple flask app ever. What it does is sets the route to the current directory, and returns 'Something'  

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1> Something </h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
```
** Above is copy and pasteable flask starter code. In your browser go to **    http://0.0.0.0:6969/

---

<br>
<br>
### This is how to make a new page at a new address.
This code makes a new page called `zack_rules`, and fills it up with content generated when that page is visited.
You do this via the `@app.route()` method.  and put your function below it.  
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1> Something </h1>'


# adding a new page
@app.route('/zack_rules')
def zack_rules():
    content = ''
    # generate some content
    for i in xrange(1000):
        content += 'zack rules! '
    return '<h1> Yo: %s </h1>' % content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
```
**Above is copy and pasteable example of how you can pass a list of strings into your webpage onto a page named 'zack_rules'.**
*http://0.0.0.0:6969/zack_rules*

---
<br>
<br>
<br>


# STEP 3:  Basic human interaction.
Feeding user submitted data from your webpage into python, doing something with that data (like feed it into your model and make a prediction), then returning an output to the web.

So what we are doing here is using our flask app to interact with new information.  Here is the a simple example of how to setup a web page with a text box that the user can use to 'pass' information into your app.

To setup this 'pipeline' we are going to  build an html form in in actual html inside of our python app.  The form looks like this.  
```python
from flask import Flask
from flask import request
app = Flask(__name__)

# our home page
@app.route('/')
def index():
    return '<h1> Something </h1>'


# create a new page
@app.route('/zack_rules')
def zack_rules():
    content = ''
    for i in xrange(1000):
        content += 'zack rules! '
    return '<h1> Yo: %s </h1>' % content


# create page with a form on it
@app.route('/submission_page')
def submission_page():
    return '''
    <form action="/word_counter" method='POST' >
        <input type="text" name="user_input" />
        <input type="submit" />
    </form>
    '''


# create the page the form goes to
@app.route('/word_counter', methods=['POST'] )
def word_counter():
    # get data from request form, the key is the name you set in your form
    data = request.form['user_input']

    # convert data from unicode to string
    data = str(data)

    # run a simple program that counts all the words
    dict_counter = {}
    for word in data.lower().split():
        if word not in dict_counter:
            dict_counter[word] = 1
        else:
            dict_counter[word] += 1
    total_words = len(dict_counter)

    # now return your results
    return 'Total words is %i, <br> dict_counter is: %s' % (total_words, dict_counter)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)


```







<br>
<br>
# Extra:  Using Flask with Bootstrap.
First we must install the dependencies.  
`pip install flask-bootstrap`  
`pip install flask-script`






<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br><br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
How to structure your flask app directory.  
```
MyProject/
|-- Python/
|   |-- bin
|   |   |-- app.py
|   |-- docs
|   |-- sessions
|   |-- tests
|   |   |-- project_tests.py
|
|-- static/
|   |-- css/
|   |   |-- bootstrap.css
|   |   |-- bootstrap.min.css
|   |   |-- my_custom_styles.css
|   |-- js/
|   |   |-- bootstrap.js
|   |   |-- bootstrap.min.js
|   |-- img/
|   |   |-- glyphicon-halflings.png
|   |   |-- glyphicon-halflings.png
|   |-- fonts/
|   |   |-- cool_font
```
This is how you install bootstrap for flask
* dont forget to run pip install flask script bootsrap thing****
Main HTTP methods:
@app.route(methods=['see options below'])
URL + method = unique response
* Get
    * reads whatever is at the end point.
* Post
* Put
* Delete
