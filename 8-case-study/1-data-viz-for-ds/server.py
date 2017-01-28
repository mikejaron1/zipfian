from flask import Flask, render_template
import pdb
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/<city>")
def viz(city):
    # pdb.set_trace()
    
    # to read file and return raw data
    #return open('data/' + city + '-6hw.csv', 'r').read()
    
    # render a template with the correct city in the d3.csv function
    return render_template('solution_nvd3.html', city=city)
    
if __name__ == "__main__":
    app.run()