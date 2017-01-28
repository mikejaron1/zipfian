from flask import Flask
from flask import render_template
import pandas as pd
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('image.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
