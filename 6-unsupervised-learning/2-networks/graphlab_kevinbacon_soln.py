#!/usr/bin/env python
# -*- coding: utf-8 -*-


# coding: utf-8

### GraphLab Graph Analytics Toolkit - Exploring the graph of American Films

# **Note: This notebook uses GraphLab Create 0.9.**
# 
# Welcome to the GraphLab Graph Analytics toolkit. In this notebook we'll use the toolkit to explore American films released between 2004 and 2013 and answer the question of whether Kevin Bacon is really the best actor to put at the center of the [Kevin Bacon game](http://en.wikipedia.org/wiki/Six_Degrees_of_Kevin_Bacon).

#### Set up and exploratory data analysis

# Before we start playing with the data, we need to import the required libraries: numpy, pandas, urllib, matplotlib, and of course, graphlab. We also tell IPython notebook and GraphLab Canvas to produce plots directly in the notebook.

# In[1]:

import numpy as np
import pandas as pd
import urllib
import matplotlib.pyplot as plt
from matplotlib import rcParams

import graphlab

from IPython.display import display
from IPython.display import Image
graphlab.canvas.set_target('ipynb')
get_ipython().magic(u'matplotlib inline')


# We'll load data on performances in American movies for the last ten years, pulled from the [Freebase](http://www.freebase.com/) API's [film/film](http://www.freebase.com/film/film?schema) and [film/performance](http://www.freebase.com/film/performance?schema=) topics. The Freebase data is crowd-sourced, so it's a bit messy, but it is freely available under the [Creative Commons license](http://creativecommons.org/licenses/by/2.5/). Our curated data live in an Amazon S3 bucket, but for this demo we'll first download the CSV file and save it locally. Please note that running this notebook on your machine *will* download the 8MB csv file to your working directory.

# In[2]:

url = 'https://s3.amazonaws.com/GraphLab-Datasets/americanMovies/freebase_performances.csv'
urllib.urlretrieve(url, filename='freebase_performances.csv')  # downloads an 8MB file to the working directory


# There are a few data preprocessing steps to do, for which we'll use an **SFrame** (for more on SFrames, see the [Introduction to SFrames](http://graphlab.com/learn/notebooks/introduction_to_sframes.html) notebook). First, we drop a superfluous column which is created because of a missing column header. Next, we drop actor names that are equal to an empty list, which is obviously an error. Finally, we add a column with 0.5 in each row, which will come in handy later for computing graph distances between actors.
# 
# After the data is clean, let's take a peek.

# In[3]:

data = graphlab.SFrame.read_csv('remote://freebase_performances.csv',
                                column_type_hints={'year': int})
data = data[data['actor_name'] != '[]']
data['weight'] = .5
data.show()


# Now we construct the graph. For the first half of our analysis we will use a bipartite graph, where actors and movies are vertices and film performances are the edges that connect actors to movies. 
#   
#   - First we get **SArrays of the unique actor and film names**. This will help us identify which vertices in our graph belong to each of these two classes.
#   - The **SGraph is directed**, so to create an undirected graph we add edges in each direction.
#   - The SGraph constructor automatically creates vertices based on the source and destination fields in the edge constructor.

# In[4]:

actors = data['actor_name'].unique()
films = data['film_name'].unique()

g = graphlab.SGraph()
g = g.add_edges(data, src_field='actor_name', dst_field='film_name')
g = g.add_edges(data, src_field='film_name', dst_field='actor_name')

print "Movie graph summary:\n", g.summary(), "\n"


# By using the **get_vertices()** and **get_edges()** methods we can verify the data was entered correctly. Note the SGraph uses directed edges, so we enter them in both directions to get an undirected graph and the correct output from the toolkits.

# In[5]:

print "Actor vertex sample:"
g.get_vertices(ids=actors).tail(5)


# In[6]:

print "Film vertex sample:"
g.get_vertices(ids=films).head(5)


# In[7]:

print "Sample edges (performances):"
g.get_edges().head(5)


# This graph can (and should) be saved so we can come back to it later without the data cleaning.

# In[8]:

# g.save('sample_graph')
# new_g = graphlab.load_graph(filename='sample_graph')


# This graph is too large to visualize, so we'll pull a small subgraph to see what the data look like.

# In[9]:

selection = ['The Great Gatsby', 'The Wolf of Wall Street']

subgraph = graphlab.SGraph()
subgraph = subgraph.add_edges(g.get_edges(dst_ids=selection),
                              src_field='__src_id', dst_field='__dst_id')
subgraph.show(highlight=selection)


# Can you guess which node is in the middle?

# In[10]:

subgraph.show(vlabel='id', highlight=selection)


#### Connected Components

# First, let's find the number of connected components. We'll do this first on our bipartite graph with both films and actors, but in the second half of this notebook we'll do it again with only actors.

# In[11]:

cc = graphlab.connected_components.create(g, verbose=False)
cc_out = cc['component_id']
print "Connected components summary:\n", cc.summary()


# There are over 2,000 components. With the 'component_size' field we can see that there is really only one very large component.

# In[12]:

cc_size = cc['component_size'].sort('Count', ascending=False)
cc_size


# Let's pull one of the smaller connected components to see if there's anything interesting. The *cc_out* object is an SFrame, which acts a lot like a Pandas DataFrame, but is on disk.

# In[13]:

tgt = cc_size['component_id'][1]
tgt_names = cc_out[cc_out['component_id'] == tgt]['__id']

subgraph = graphlab.SGraph()
subgraph = subgraph.add_edges(g.get_edges(src_ids=tgt_names),
                              src_field='__src_id', dst_field='__dst_id')

film_selector = subgraph.get_vertices(ids=films)['__id']
subgraph.show(vlabel='id', highlight=film_selector)


# This component corresponds to a handful of Japanese anime series. To help ourselves out later, we'll also pull out the names of the actors in the giant connected component. Here we use the **SFrame.filter_by method** because we're looking for matches against a set of names.

# In[14]:

big_label = cc_size['component_id'][0]
big_names = cc_out[cc_out['component_id'] == big_label]
mainstream_actors = big_names.filter_by(actors, column_name='__id')['__id']


#### The Kevin Bacon game

# OK, let's play the Kevin Bacon game. First, let's see what movies he's been in over the last decade...

# In[15]:

bacon_films = g.get_edges(src_ids=['Kevin Bacon'])

subgraph = graphlab.SGraph()
subgraph = subgraph.add_edges(bacon_films, src_field='__src_id',
                              dst_field='__dst_id')
subgraph.show(vlabel='id', elabel='character', highlight=['Kevin Bacon'])


# ... and with whom Kevin Bacon has co-starred. Hover over nodes with the mouse to see the labels.

# In[16]:

subgraph = graphlab.SGraph()

for f in bacon_films['__dst_id']:
    subgraph = subgraph.add_edges(g.get_edges(src_ids=[f], dst_ids=None),
                                  src_field='__src_id', dst_field='__dst_id')
    
subgraph.show(highlight=list(bacon_films['__dst_id']), vlabel='__id', vlabel_hover=True)


# We can find the shortest path distance between every other vertex and Mr. Bacon. Keep in mind that this is a bipartite graph, so movies will have half distances from our target and actors will have whole number distances from the target.

# In[17]:

sp = graphlab.shortest_path.create(g, source_vid='Kevin Bacon', weight_field='weight', verbose=False)
sp_graph = sp['graph']


# The computation is very quick. And we can now get the distance from Kevin Bacon to any other actor or movie. Kevin has a distance of 0, of course, and the films he's been in lately are all at distance 0.5. His co-stars in those movies are at distance 1, so they have Kevin Bacon number of 1.
# 
# Querying this for another actor is very fast because the result is already computed. Robert De Niro, for example, has a Kevin Bacon number (for the last decade) of 2. There are many paths of length two between these two actors, and 'get_path' will plot one of them.
# 
# If we go back to the connected component of Japanese films and pick one of the actors, Yuzuru Fujimoto, we see he has an infinite Kevin Bacon number over the last decade, which fits the earlier result that Yuzuru is in a different component. If we try to find the path between these two actors, we get an astronomically high number, indicating there is no path.

# In[18]:

path = [x[0] for x in sp.get_path('Robert De Niro', show=True, highlight=list(films))]

query = sp_graph.get_vertices(ids='Yuzuru Fujimoto')
query.head()


# Who else do we want to check?? Because we're not using name disambiguation, we should check first to make sure a suggestion is in the list of actors.

# In[19]:

target = 'Lydia Fox'
target in actors

path = [x[0] for x in sp.get_path(target, show=True, highlight=list(films))]


# To visualize the distribution of Kevin Bacon distances, we narrow the result down to only mainstream actors (i.e. those in the same connected component as Mr. Bacon). The most common Kevin Bacon number is 3. Note there are still some half distances in this result; this is because we're working with messy data. The messiness comes from three sources:
# 
#   1. Some names are both actors and films (e.g. Secretariat)
#   2. Freebase is crowd-sourced data with plenty of mistakes
#   3. For the demo we're using proper names as vertex IDs, so when there is an overlapping name, we have problems. In production we would use a vertex ID known to be unique.
#   
# Before we plot the distribution of Kevin Bacon distances, we define the *clean_plot* function to make our matplotlib output look nice.

# In[20]:

bacon_sf = sp_graph.get_vertices(ids=mainstream_actors)

def clean_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.patch.set_facecolor('0.92')
    ax.set_axisbelow(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
rcParams['figure.figsize'] = (10, 8)
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
    
fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(list(bacon_sf['distance']), bins=30, color='cornFlowerBlue')
ax.set_xlabel('Kevin Bacon distance')
ax.set_ylabel('Frequency')
clean_plot(ax)
fig.show()


#### Why Kevin Bacon?

# Why is Kevin Bacon the target of the Kevin Bacon game? Why not Robert De Niro or Dennis Hopper, both of whom have been in a lot of movies. In fact, let's figure out just how central Kevin Bacon is, and who might be a better center for the Kevin Bacon game. First, let's find out who's been in the most movies. We'll start with some good guesses about prolific actors.

# First, let's find out the number of movies for each actor. To do this we use the **SGraph.triple_apply** method, which loops over edges in the graph. For each edge, the method applies a user-specified function that changes some
# attributes of the vertices incident to the edge. In this case, there are three steps, encapsulated in the **get_degree** function. 
# 
# 1. make a copy of the graph.
# 2. make a new vertex attribute *in_degree* to hold the result (i.e. the number of movies for each actor and the number of actors for each film).
# 3. run the triple_apply function, which applies the **count_in_degree** function to each edge of the graph. In this case we add 1 to the in-degree of each edge's destination node. Because we added edges in both directions, this is sufficient to compute the degree.

# In[21]:

def count_in_degree(src, edge, dst):
    dst['in_degree'] += 1
    return (src, edge, dst)

def get_degree(g):
    new_g = graphlab.SGraph(g.vertices, g.edges)
    new_g.vertices['in_degree'] = 0
    return new_g.triple_apply(count_in_degree, ['in_degree']).get_vertices()

degree = get_degree(g)


# In[22]:

comparisons = ['Kevin Bacon', 'Robert De Niro', 'Dennis Hopper', 'Samuel L. Jackson']
degree.filter_by(comparisons, '__id').sort('in_degree', ascending=False)


# At least for the last decade, Samuel L. Jackson seems to be a better candidate than
# Kevin Bacon. But we're left with several questions: were our candidate guesses good? who are the top actors overall?
# how does Kevin Bacon stack up? To answer these we still need to separate the actors from
# the films, which we can do by filtering with the actors list.

# In[23]:

actor_degree = degree.filter_by(actors, '__id')


# In[24]:

fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(list(actor_degree['in_degree']), bins=50, color='cornFlowerBlue')
ax.axvline(degree['in_degree'][degree['__id'] == 'Kevin Bacon'], color='red', lw=2, label='Kevin Bacon')
ax.set_ylabel('Frequency')
ax.set_xlabel('Number of movies')
ax.legend()
clean_plot(ax)
fig.show()


# This confirms that Kevin Bacon---with 21 movies in the last decade---is indeed prolific, but he's far from the top. So who is at the top??

# In[25]:

actor_degree.sort('in_degree', ascending=False)


# In[26]:

print "** Danny Trejo**"
display(Image(url='https://s3.amazonaws.com/GraphLab-Datasets/americanMovies/Danny-Trejo.jpg'))
print "Source: Glenn Francis http://www.PacificProDigital.com"


# *Danny Trejo* is the winner!! But number of movies is a crude measure of centrality. John Cena, for example, is a professional wrestler, whose 70 movies are almost entirely WWE performances. Let's also find the mean shortest path distance for our selected comparisons, which is the average shortest path distance from each source actor to all other actors.

# In[27]:

# Add top-degree actors to the comparison list
comparisons += ['Danny Trejo', 'Frank Welker', 'John Cena']


# In[28]:

# # Make a container for the centrality statistics
mean_dists = {}

# # Get statistics for Kevin Bacon - use the already computed KB shortest paths
mean_dists['Kevin Bacon'] = bacon_sf['distance'].mean()


## Get statistics for the other comparison actors
for person in comparisons[1:]:

    # get single-source shortest paths
    sp2 = graphlab.shortest_path.create(g, source_vid=person,
                                        weight_field='weight',
                                        verbose=False)
    sp2_graph = sp2.get('graph')
    sp2_out = sp2_graph.get_vertices(ids=mainstream_actors)

    # Compute some statistics about the distribution of distances
    mean_dists[person] = sp2_out['distance'].mean()

    # Show the whole distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(list(sp2_out['distance']), bins=30, color='cornFlowerBlue')
    ax.axvline(mean_dists[person], color='black', lw=3,
               label="mean={}".format(round(mean_dists[person], 2)))
    ax.axvline(mean_dists['Kevin Bacon'], color='black', lw=2, ls='--',
               label="Bacon mean={}".format(round(mean_dists['Kevin Bacon'], 2)))
    ax.legend()
    ax.set_xlabel('{} distance'.format(person))
    ax.set_ylabel('Frequency')
    clean_plot(ax)

    fig.show()


# In addition to having the largest degree in the whole graph, Danny Trejo has a smaller mean shortest path distance to the rest of the mainstream_actor graph than Kevin Bacon, and there are far fewer people who are 4 degrees of separation away from him. He seems to be a much better candidate for the center of the Kevin Bacon game.
# 
# However, there are many other measures of centrality or importance in a graph. Some of these require converting our bipartite graph into an actor network where edges join two actors who have been in a movie together.

#### Computing the actor network

# To compute other interesting statistics about the network of actors, it is useful to eliminate the film vertices, so that actor vertices share an edge of the two actors co-starred in a movie. If A is the adjacency matrix for the bipartite graph where actors are rows and movies are columns, the social network (with weighted edges) is computed by AA^T. For now this is done with numpy and Pandas. Because this operation is memory intensive, we first pull out the subset of movies from 2013.

# In[35]:

## Pull out the data for 2013 films
year_data = data[data['year'] == 2013]

year_actors = graphlab.SFrame({'actor': year_data['actor_name'].unique()})
year_actors = year_actors.add_row_number()

year_films = year_data['film_name'].unique()


# Next we construct the A matrix and multiply it by itself. This is a computational bottleneck that takes some time and is the reason we subset the data down to just 2013 release year movies.

# In[36]:

A = pd.DataFrame(np.zeros((len(year_actors), len(year_films)), dtype=np.int),
                 columns=year_films, index=year_actors['actor'])

for row in year_data:
    A[row['film_name']][row['actor_name']] = 1


# In[37]:

A = A.values
adjacency = np.triu(np.dot(A, A.T))


# Finally, we construct a new graph which is the actor network, where two actors are connected if they've been in a movie together. The edge attribute 'count' is the number of movies shared by two actors in 2013 - with a handful of exceptions, this is 1 for all actor pairs (in the 2013 data).

# In[38]:

year_actors


# In[39]:

edge_idx = np.nonzero(adjacency)
sf_edge = graphlab.SFrame({'idx_source': edge_idx[0], 
                           'idx_dest': edge_idx[1],
                           'weight': adjacency[edge_idx]})

sf_edge = sf_edge.join(year_actors, on={'idx_source': 'id'}, how='left')
sf_edge.rename({'actor': 'actor1'})

sf_edge = sf_edge.join(year_actors, on={'idx_dest': 'id'}, how='left')
sf_edge.rename({'actor': 'actor2'})

sf_edge.remove_column('idx_dest')
sf_edge.remove_column('idx_source')

net = graphlab.SGraph()
net = net.add_edges(sf_edge, src_field='actor1', dst_field='actor2')
net = net.add_edges(sf_edge, src_field='actor2', dst_field='actor1')


# In[40]:

print "Sample actor edges:"
net.edges


#### Connected components (again)

# We can also find the connected components in the actor network.

# In[41]:

cc = graphlab.connected_components.create(net)
cc_out = cc.get('component_id')

print "Connected component summary:"
cc.summary()


# Again, there are is one dominant component with "mainstream" actors. As with the bipartite graph, let's isolate and explore some of the smaller components.

# In[42]:

cc_size = cc['component_size'].sort('Count', ascending=False)
cc_size


# To keep things simple going forward, we'll work with only the big connected component.

# In[43]:

big_label = cc_size['component_id'][0]
big_names = cc_out[cc_out['component_id'] == big_label]
mainstream_actors = big_names.filter_by(actors, column_name='__id')['__id']

mainstream_edges = net.get_edges(src_ids=mainstream_actors)
net = graphlab.SGraph()
net = net.add_edges(mainstream_edges, src_field='__src_id', dst_field='__dst_id')

net.summary()


#### Back to Kevin Bacon

# Kevin Bacon appears to have been eclipsed by many actors as the center of the Kevin Bacon game over the past decade, but we used very crude graph metrics to determine this. In this smaller network of 2013 actors we can compute more sophisticated things to measure centrality. We'll start with vertex degree once again, but now an actor's degree is the number of actors who share a movie with him (rather than the number of movies done by the actor as in the bipartite graph).
# 
# We'll use an SFrame---initialized with the actor degree---to store the results.

# In[44]:

centrality = get_degree(net)


# Triangles in a graph are complete subgraphs with only three vertices. The number of triangles to which an actor belongs is a measure of the connectivity of his or her social network.

# In[45]:

tc = graphlab.triangle_counting.create(net)
print "Triangle count summary:\n", tc.summary()

centrality = centrality.join(tc['triangle_count'], on='__id', how='left')


# Pagerank is a popular method for calculating the "importance" of nodes in a graph.

# In[46]:

pr = graphlab.pagerank.create(net, verbose=False)
print "Pagerank summary:\n", pr.summary()

centrality = centrality.join(pr['pagerank'], on='__id', how='left')
centrality.sort('pagerank', ascending=False)


# James Franco crushes the competition with pagerank. If we plot the histogram of pagerank, we can see just how far ahead Mr. Franco really is.

# In[47]:

idx_bacon = centrality['__id'] == 'Kevin Bacon'
idx_franco = centrality['__id'] == 'James Franco'

fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(list(centrality['pagerank']), bins=20, color='cornFlowerBlue')
ax.set_ylabel('Frequency')
ax.set_xlabel('Pagerank')
ax.axvline(centrality['pagerank'][idx_franco], color='red', lw=2,
           label='James Franco')
ax.axvline(centrality['pagerank'][idx_bacon], color='black', lw=2, ls='--',
           label='Kevin Bacon')
ax.legend()
clean_plot(ax)
fig.show()


# Finally, another very common way to measure the centrality of a vertex is the mean distance from the vertex to all other nodes. This is a relatively expensive thing to compute, so we'll find it just for the top of our leader board.

# In[48]:

centrality = centrality.sort('pagerank', ascending=False)

mean_dists = [int(1e12)] * centrality.num_rows()

for i in range(10):
    a = centrality[i]['__id']
    sp = graphlab.shortest_path.create(net, source_vid=a, verbose=False)
    sp_out = sp['distance']
    mean_dists[i] = sp_out['distance'].mean()

centrality['mean_dist'] = mean_dists
centrality.sort('mean_dist')


# We have a three way tie, between *Anthony Mackie*, *James Franco* , and *Paul Rudd*.

# In[49]:

display(Image(url='https://s3.amazonaws.com/GraphLab-Datasets/americanMovies/Anthony-Mackie.jpg'))
print "Source: David Shankbone http://flickr.com/photos/27865228@N06/4543207958"
display(Image(url='https://s3.amazonaws.com/GraphLab-Datasets/americanMovies/James-Franco.jpg'))
print "Source: Vanessa Lua http://www.flickr.com/photos/vanessalua/6286991960/"
display(Image(url='https://s3.amazonaws.com/GraphLab-Datasets/americanMovies/Paul-Rudd.jpg'))
print "Source: Eva Rinaldi http://www.flickr.com/photos/evarinaldiphotography/11024133765/" 


# TL; DR: Danny Trejo is the real Kevin Bacon of the last decade, while Paul Rudd, Anthony Mackie and James Franco took over the most central spot for 2013.
# 
# (Looking for more details about the modules and functions? Check out the <a href="/products/create/docs/">API docs</a>.)

# In[ ]:



