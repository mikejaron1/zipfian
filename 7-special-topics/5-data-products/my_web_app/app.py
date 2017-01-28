from flask import Flask, request
import random
import requests
import ipdb
import pickle


# Initialize your app and load your pickled models.
#================================================
# init flask app
app = Flask(__name__)

# load the model you pickled in build_model.py
model = pickle.load(open('data/my_model.pkl', 'r'))

# load the vectorized you pickled in build_model.py
tfidf = pickle.load(open('data/my_vectorizer.pkl', 'r'))




# Homepage with form on it.
#================================================
@app.route('/')
def index():
    return '''
    <form action="/predict" method='POST' >
        <input type="text" name="user_input" />
        <input type="submit" />
    </form>
    '''

# Once submit is hit, pass info into model, return results.
#================================================
@app.route('/predict', methods=['POST'])
def predcit():

    # get data from request form
    data = request.form['user_input']

    # convert data from unicode to string
    data = str(data)

    # vectorize our new data
    features = tfidf.transform([data])

    # make prediction based on new data
    pred = model.predict(features)

    # return a string format of that prediction to the html page
    return str(pred)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
