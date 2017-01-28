from flask import Flask, request
import random
import requests
import ipdb
import pickle
app = Flask(__name__)


@app.route('/')
def index():
    return '''go to your /get_datafile page
    '''

# Using requests.get() to reterive a url. 
#============================================
@app.route('/get_datafile')
def get_datafile():
    # use requests.get(some-url)
    # when page is loaded, this command runs and gets the data and passes it into data
    data = requests.get('http://www.reddit.com/r/Infographics.json')

    # become fimilar with the request object, it is your key to the world.     
    print type(data)
    
    # ipdb.set_trace()
    return str(data.content)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
