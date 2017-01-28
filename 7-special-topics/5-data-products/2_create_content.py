from flask import Flask
from flask import request
app = Flask(__name__)

# our home page
#============================================
@app.route('/')
def index():
    return '<h1> Something </h1>'


# /zack_rules page  
#============================================
# Writes zack rules 1000 times then sends that string to html
@app.route('/zack_rules')
def zack_rules():
    content = ''
    for i in xrange(1000):
        content += 'zack rules! '
    return '<h1> Yo: %s </h1>' % content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
