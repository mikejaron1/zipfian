
# coding: utf-8

# In[1]:

import networkx as nx
# import seaborn as sns


# In[8]:

### A HELPER FUNCTION TO SORT DICTS
def sortdict(adict):
    import operator
    sorted_dict = sorted(adict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_dict


# In[9]:

G = nx.read_edgelist('data/ross-alumni-edge-table.csv', delimiter=",")
G = nx.read_edgelist('data/find-yourself.csv', delimiter=",")
G = G.to_undirected()


# In[10]:

# NUMBER 1,2,3
figsize(20,20)
nx.draw(G, with_labels=False)


# In[11]:

# NUMBER 4,5
pos = nx.spring_layout(G,dim=2, k=0.1)
nx.draw(G,pos,with_labels=False)


# In[12]:

### NUMBER 6,7

### FIND THE NODES WITH THE HIGHEST DEGRESS
deg = nx.degree(G)

### PLOT A HISTOGRAM OF THE NODE DEGREES
### ### WHAT DOES THIS TELL US ABOUT THE GRAPH
figsize(4,4)
plt.hist(deg.values());
plt.xlabel("DEGREE aka NUMBER OF CONNECTIONS")
plt.ylabel("Number of nodes in bin")
# plt.bar(left=deg.values(), height=deg.values());


# In[13]:

### NUMBER 8,9
figsize(10,10)
pos = nx.spring_layout(G,dim=2, k=0.1,scale=0.25)
nx.draw(G,pos,with_labels=False,node_size=deg.values())


# In[24]:

### NUMBER 10
ccoeff = nx.clustering(G)
print type(ccoeff)
# figsize(5,5)
# plt.hist(ccoeff.values());



# In[31]:

### NUMBER 11
figsize(10,10)
deg = nx.degree(G)
deg

# pos = nx.spring_layout(G,dim=2, k=0.1,scale=0.25)

nx.draw(G,pos,with_labels=False, node_size=deg.values(), node_color=deg.values(), cmap=plt.cm.Accent)


# In[16]:

### NUMBER 12
figsize(10,10)
pos = nx.spring_layout(G,dim=2, k=0.1,scale=0.25,iterations=5000)
nx.draw_networkx_nodes(G,pos,with_labels=False,node_size=deg.values(), node_color=ccoeff.values(),cmap=plt.cm.Spectral)
nx.draw_networkx_edges(G,pos,alpha=0.1)


# In[17]:

# # pos = nx.spring_layout(G,dim=2, k=0.1,scale=0.25)
# pos = nx.circular_layout(G)
# nx.draw_networkx_nodes(G,pos,with_labels=False,node_size=deg.values(), node_color=ccoeff.values(),cmap=plt.cm.Spectral)
# nx.draw_networkx_edges(G,pos,alpha=0.1)




### NUMBER 1
G = nx.read_edgelist('data/find-yourself.csv', delimiter=",")
G = G.to_undirected()


# In[15]:

### NUMBER 2)
luke = nx.shortest_path(G,'Jonny Lee', 'Luke DeSario')
print "LUKE", luke

jon = nx.shortest_path(G,'Jonny Lee', 'Jonathan Dinu')
print "Jon", jon

gio = nx.shortest_path(G,'Jonny Lee', 'Giovanna Thron')
print "Gio", gio

ryan = nx.shortest_path(G,'Jonny Lee', 'Ryan Orban')
print "Ryan", ryan

pato = nx.shortest_path(G,'Jonny Lee', 'Patricio Ovalle')
print "Pato", pato

kevin = nx.shortest_path(G,'Jonny Lee', 'Kevin Joseph')
print "Kevin", kevin

cy = nx.shortest_path(G,'Jonny Lee', 'Cyrus Buffum')
print "Cy", cy

who = ['Clavel Salinas','Abi Kelly', 'Charley Cohen', 'Mats Andersson', 'Nicolas Cardyn', 'Dashiel Marder']

for p in who:
    sp = nx.shortest_path(G, 'Jonny Lee', p)
    print p, ":" ,sp


# In[ ]:

### NUMBER 3
deg = nx.degree(G)
sortdict(deg)
# (u'Athanasios George Polychronopoulos', 305)


# In[14]:

### NUMBER 4
figsize(20,20)
gio_ego = nx.ego_graph(G,'Jonny Lee',radius=3)
cc = nx.closeness_centrality(gio_ego)
ac = nx.average_clustering(gio_ego)
nx.draw(gio_ego,node_color=cc.values(),cmap=plt.cm.Blues)