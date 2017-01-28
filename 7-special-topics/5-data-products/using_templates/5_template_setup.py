from flask import Flask
from flask import render_template
import pandas as pd
app = Flask(__name__)


@app.route('/')
def hello_world():
    df = pd.read_csv('some.csv')
    df = df.set_index('Unnamed: 0')
    px = df['px'].values
    years = df['year'].values
    data = zip(px,years)
    return render_template('my_template.html', data=data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
