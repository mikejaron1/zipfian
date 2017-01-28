
# coding: utf-8

# In[2]:

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import toolz
import pandas as pd
import os

get_ipython().magic(u'pylab inline')


# ### filter original data
# ```python
# for idx in pd.read_csv("data/twitter/circles.txt", delimiter='\t', header = None).T.as_matrix():
#    !cp data/twitter/$idx[0]* data/twitter/subset/
# ```

# ```python
# base = "data/twitter/subset"
# 
# fw = open("data/twitter/full.txt", 'w') 
# ```

# ```python
# for i in os.listdir(os.getcwd() + '/' + base):
#     with open(base + "/" + i, 'r') as fh:
#         node = i.split('.')[0]
#         edges = set()
#         
#         for line in fh:
#             nodes = line.split(' ')
#             edges.add(nodes[0].strip())
#             edges.add(nodes[1].strip())
#         
#         for edge in edges:   
#             fw.write("%s %s\n" % (node, edge))
# fw.close()
# ```

# In[3]:

G = nx.read_edgelist('data/twitter.txt', delimiter=' ', create_using=nx.DiGraph())


# In[4]:

G.number_of_edges()


# In[5]:

G.number_of_nodes()


# In[6]:

plt.hist(G.in_degree().values(), bins=100);


# In[7]:

plt.hist(G.out_degree().values(), bins=100);


# In[8]:

G.in_degree()


# In[9]:

def sort_dict(d):
    return sorted(d.items(),  key= lambda x: x[1], reverse=True)


# In[10]:

np.max(G.in_degree().values())


# In[11]:

sort_dict(G.in_degree())


# In[12]:

sort_dict(nx.centrality.in_degree_centrality(G))


# In[31]:

sort_dict(nx.centrality.betweenness_centrality(G))


# In[30]:

sort_dict(nx.centrality.closeness_centrality(G))


# In[14]:

sort_dict(nx.centrality.eigenvector_centrality(G))


# In[15]:

[ node for node in G.nodes() if G.out_degree(node) == 50 ]


# In[16]:

user = '643653'


# In[17]:

user


# In[18]:

G.out_edges(user)


# In[19]:

following = set([ n[1] for n in G.out_edges(user) ])


# In[20]:

total_conns = []

for node in following:
    total_conns += [ n[1] for n in G.out_edges(node) ]


# In[21]:

total_conns


# In[22]:

cnts = toolz.itertoolz.frequencies(total_conns)
uniques = [ (node, cnts[node]) for node in cnts if node not in following ]


# In[23]:

len(uniques)


# In[24]:

len(cnts)


# In[25]:

len(following)


# In[26]:

uniques


# In[27]:

sorted(uniques, key= lambda x: x[1], reverse=True)


# In[28]:

sorted(uniques, key= lambda x: x[1], reverse=True)[:10]


# In[29]:

plt.hist([ u[1] for u in uniques ], bins=100);
plt.vlines(20, 1, 100, color="red")

