from flask import Flask
from flask import request
app = Flask(__name__)

# our home page
#============================================
@app.route('/')
def index():
    return '<h1> Something </h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
