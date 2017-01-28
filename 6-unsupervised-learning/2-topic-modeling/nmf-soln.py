from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import matplotlib.pyplot as plt

import plotly.plotly as py
from plotly import graph_objs

# 1. Load NYT articles database
articles = pd.read_pickle('data/articles.pkl')

# Pre-processing. Find empty articles and remove rows with empty content
articles = articles[articles['content'].apply(lambda x: len(x)) != 0]
content = articles['content']

# 2, 3. Finding top features with top topics and top words
n_features = 5000
n_topics = 15
n_top_words = 10

vect = TfidfVectorizer(stop_words='english', max_features=n_features)
nmf = NMF(n_components=n_topics)

bags = vect.fit_transform(content)
W = nmf.fit_transform(bags)
H = nmf.components_


# 4. Explore content and print n_top_words from each
feature_words = vect.get_feature_names()
for i in xrange(W.shape[1]):
    print 'Article %d' % i
    print articles['headline'][np.argsort(W[:,i])[:-n_top_words:-1]]

for i in xrange(H.shape[0]):
    print 'Topic %d' % i
    print [feature_words[j] for j in np.argsort(H[i, :])[:-n_top_words:-1]]

# put the pieces together for all topics to get matplotlib code and 
# compare with plotly.

# def topic_parse(H, n_topics):
#     topics_dicts = []

#     for i in xrange(n_topics):
#         # n_top_words of keys and values
#         keys, values = zip(*sorted(zip(feature_words, H[i]), key = lambda x: x[1])[:-n_top_words:-1])
#         val_arr = np.array(values)
#         norms = val_arr / np.sum(val_arr)
#         topics_dicts.append(dict(zip(keys, np.rint(norms* 300))))
#     return topics_dicts

# topic_dicts = topic_parse(H, n_topics)

# plt.bar(np.arange(len(H[1,:])),H[1,:])
# plt.show()

# need to sign in with your own credentials here
# py.sign_in('user_name', "api_key')

# 5. Add title to each latent features with topics they represent
trace1 = graph_objs.Bar(
    x=feature_words,
    y=H[0,:],
    name='Finance'
)
trace2 = graph_objs.Bar(
    x=feature_words,
    y=H[1,:],
    name='Football'
)

data = graphobjs.Data([trace1, trace2])

layout = graphobjs.Layout(
    title='Word Distributions for Topics of the NYT',
    barmode='group',
    xaxis=XAxis(showticklabels=False, title="Words"),
    yaxis=YAxis(title="Relevance") )
fig = graphobjs.Figure(data=data, layout=layout)
#plot_url = py.plot(fig, filename='nyt_word_distributions', auto_open=False)
py.iplot(fig, filename='grouped-bar')

traces = []
for section in data_f['section_name'].unique():
    trace = dict(
        type='scatter',
        mode= 'markers',
        x=W_2[:,0][np.array(data_f['section_name'] == section)],
        y=W_2[:,1][np.array(data_f['section_name'] == section)],
        text = list(heads[data_f['section_name'] == section]),
        opacity = 0.8,
        showlegend= True,
        name = section
    )
    
    traces.append(trace)
    
x_axis = dict(title='Politics')

y_axis = dict(title='Leisure')

layout = dict(
    title='NYT Projected into 2D Topic Space',   
    xaxis=x_axis, # set x-axis style
    yaxis=y_axis # set y-axis style
)

fig = dict(data=traces, layout=layout)
#plot_url = py.plot(fig, filename='nyt_2d_topic',auto_open=False)
py.iplot(fig, validate=False)

# Choose topics 'Arts', 'Business Day' and 'Sports'
# sub_sect = data[data.section_name.isin(['Arts', 'Business Day', 'Sports'])]['content']
# subselect section for 3 dimensional scatterplot. 
small_data = vectorizer.fit_transform(data_f.content)
# 3 topics
nmf_3 = NMF(n_components=3)
W_3 = nmf_3.fit_transform(small_data)
# nmf.fit(bags)
# nmf.transform(bags)
H_3 = nmf_3.components_


# Colors for the dots in the scatter plot
traces = []
colors = ["#C659CB",
"#71C44D",
"#C25037",
"#9BA9C1",
"#454036",
"#AE4D76",
"#83C9A3",
"#D1C046",
"#6A62AC",
"#AA8D5C"]

for i, section in enumerate(data_f['section_name'].unique()):
    indices = np.array(data_f['section_name'] == section)
    x = W_3[:,0][indices] 
    y = W_3[:,1][indices] 
    z = W_3[:,2][indices] 
    trace = dict(
        type='scatter3d',
        opacity = 0.8,
        showlegend= True,
        name= section,
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            color= colors[i],
            size=10            
        )
    )
    
    traces.append(trace)
  
layout = dict(title='NYT articles projected into 3D Topic Space')

fig = dict(data=traces, layout=layout)

# Print axes topics since they don't work in plotly
print "X is International Politics"
print "Y is Sports"
print "Z is US Government"

from IPython.display import HTML

# manually add legends with HTML tags
matching = zip(colors, data_f['section_name'].unique())
s = "<table style=\"position: relative; bottom: 440px; left: 750px; margin-bottom:-300px\"><td>Color</td><td>Section</td>"

for color, name in matching:
    s += "<tr><td style=\"background: %s\"></td><td>%s</td></tr>" % (color, name)

s += "</table>"

#plot_url = py.plot(fig, filename='nyt_3d_topic', validate=False, auto_open=False)
HTML(py.iplot(fig, validate=False).data + s)
