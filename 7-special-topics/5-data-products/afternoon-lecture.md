# Using Jinja2 templates.

**Python + HTML = Jinja**  
A Jinja2 template is an some.html page that can speak both python and html.
#### To get started
1. `easy_install Jinja2` or `pip install Jinja2`

This is how you have to setup your files when using html templates with flask.
```
MyProject/
|-- my_app/
|   |-- app.py
|   |-- build_model.py
|   |-- data
|   |   |-- my_model.pkl
|   |   |-- my_vectorizer.pkl
|   |-- templates
|   |   |-- homepage.html
|   |   |-- different_page.html
|   |-- static
|   |   |-- mycss.css
|   |   |-- image.png
```
Here is an example of how to setup your app.py file if you want to work with an html template file.  

*MyProject/my_app/**app.py***
```python
#MyProject/my_app/app.py
from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('python_in_html.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
```
<br>
Above is where you call the python_in_html.html file, and inside the templates folder is where that file lives.  In that file is where you can type **both valid html and valid python code.**
<br>
<br>
*MyProject/my_app/templates/**python_in_html.html**  *
```html
<!doctype html>
<head>
<link rel=stylesheet type=text/css href="{{ url_for('static', filename='my_css.css') }}">
</head>
<body>
    {% for i in range(100) %}
        <div class="classname">
            <p> BISH # {{ i }} </p>
        </div>
    {% endfor %}
</body>

```
<br>

*Notice how we referenced the **my_css.css** file.*

That css file must live in the *static* folder.  
*MyProject/my_app/static/**my_css.css**  *
```css
body {
    background-color: #000000;
}

.classname {
    text-align:center;
    background:#00FFFF;
    width: 10%;
    float:left;
}

.classname:hover {
    background:#000000;
}
```
---
<br>
<br>
<br>
## Passing variables from python into an html template.
In the example below...  
1.  I use pandas to read in a some.csv data,  
2.  Then extract the values i want to display,  
3.  Then return them with passing_variables_into_html.html



```python
from flask import Flask
from flask import render_template
import pandas as pd
app = Flask(__name__)

@app.route('/')
def hello_world():
    # 1 Import data with pandas
    df = pd.read_csv('some.csv')

    # 2 Extract values
    df = df.set_index('Unnamed: 0')
    px = df['px'].values
    years = df['year'].values
    data = zip(px,years)

    # 3 Return them with passing_variables_into_html.html
    return render_template('passing_variables_into_html.html', data=data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)

```

Now, we can access that 'data' variable from within our passing_variables_into_html.html


*MyProject/my_app/templates/**passing_variables_into_html.html**  *
```html
{% for px, year in data %}
    <table border="1px solid black" style="width:300px">
    <tr>
    <td>{{ px }}</td>
    <td>{{ year }}</td>
    </tr>
{% endfor %}
```
