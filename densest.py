import numpy as np
import time

"""
Information on graphs used and results:
- graph1: nodes=4039 edges=88234 (Social circles: Facebook) 
    => (density, nodes) = (77.34653465346534, 202) in 0.2093799114227295 seconds
- graph2: nodes=54573 edges=498202 (Graph Embedding with Self Clustering: Deezer) 
    => (density, nodes) = (16.156670746634028, 7353) in 1.421769380569458 seconds
- graph3: /!\ sanityCheck: nodes=425957 /!\ nodes=317080 edges=1049866 (DBLP collaboration network and ground-truth communities)
    => (density, nodes) = (56.5, 114) in 4.087985277175903 seconds
- graph4: nodes=1696415 edges=11095298 (Autonomous systems by Skitter) 
    => (density, nodes) = (89.18097447795823, 431) in 36.553035497665405 seconds
- graph5: /!\ sanityCheck: nodes=4036538 /!\ nodes=3997962 edges=34681189 (LiveJournal social network and ground-truth communities) 
    => (density, nodes) = (190.98445595854923, 386) in 127.30750441551208 seconds
source: http://snap.stanford.edu/data/index.html
"""


def sanityCheck(file_name, format):
  path=file_name+'.'+format
  f = open(path, 'r')
  line = f.readline()
  nodes=0
  edges=0
  while line !='':
    edges+=1
    try:
      [i, j] = line.split()
    except ValueError:
      [i, j] = line.split(',')
    i, j = int(i), int(j)
    if i > nodes:
      nodes = i
    if j>nodes:
      nodes = j
    line = f.readline()
  return nodes+1, edges

#print(sanityCheck('graph1', 'txt'))
#print(sanityCheck('graph2', 'csv'))
#print(sanityCheck('graph3', 'txt'))
#print(sanityCheck('graph4', 'txt'))
#print(sanityCheck('graph5', 'txt'))


def initGraph(file_name, format, nb_nodes):
  """
  Given a file containing all edges of a graph, saves its adjacency list to
  'adj_#file_name#.npy' and an array containing the degree of each vertex to
  'degrees_#file_name#.npy'.

  Input:
  - file_name (str): the name of the file containing the edges
  - format (str): the format (txt, csv)
  - nb_nodes (int): the number of nodes in the graph (either given by the
  source or by the function sanityCheck)
  """
  path=file_name+'.'+format
  adj = np.empty(nb_nodes, dtype=object)
  degrees = np.zeros(nb_nodes, dtype=int)
  for i in np.ndindex(adj.shape): adj[i] = []
  f = open(path, 'r')
  line = f.readline()
  while line!='':
    try:
      [i, j] = line.split()
    except ValueError:
      [i, j] = line.split(',')
    i, j = int(i), int(j)
    degrees[i]+=1
    degrees[j]+=1
    adj[i].append(j)
    adj[j].append(i)
    line = f.readline()
  np.save('adj_'+file_name, adj)
  np.save('degrees_'+file_name, degrees)
  return

#initGraph('graph1', 'txt', 4039)
#initGraph('graph2', 'csv', 54573)
#initGraph('graph3', 'txt', 425957)
#initGraph('graph4', 'txt', 1696415)
#initGraph('graph5', 'txt', 4036538)


def greedy(file_name, nb_nodes, nb_edges):
  """
  Implementation of the algorithm seen in class that tries
  to find a densest subgraph.

  Input:
  - file_name (str): the name of the file containing the edges
  - nb_nodes (int): the number of nodes in the graph (either given
  by the source or by the function sanityCheck)
  - nb_edges (int): the number of edges in the graph (either given
  by the source or by the function sanityCheck)

  Output:
  - density (float): the density of the subgraph found
  - subgraph_nodes (int): the number of nodes in the subgraph found
  """
  #Initilization
  adj = np.load('adj_'+file_name+'.npy', allow_pickle=True)
  degrees = np.load('degrees_'+file_name+'.npy', allow_pickle=True)
  erased = np.zeros(nb_nodes, dtype=bool)
  vertices = np.empty(nb_nodes, dtype=object)
  for i in np.ndindex(vertices.shape): vertices[i] = []
  locations = np.zeros(nb_nodes, dtype=int)
  mind = nb_nodes
  for i in range(nb_nodes):
    vertices[degrees[i]].append(i)
    locations[i]=len(vertices[degrees[i]])-1
    if degrees[i]<mind:
      mind = degrees[i]
  n = nb_nodes
  m = nb_edges
  density = m/n
  subgraph_nodes=n

  while n != 0:
    imin = vertices[mind].pop() #retrieve a vertex of minimal degree
    erased[imin] = 1
    neighbors = adj[imin]
    backwards=False
    #update locations, vertices, m, degrees, erased
    for j in neighbors:
      if(not(erased[j])):
        m -= 1
        d = degrees[j]
        loc = locations[j]
        length = len(vertices[d])
        vertices[d][loc], vertices[d][length-1] = vertices[d][length-1], vertices[d][loc]
        locations[vertices[d][loc]] = loc
        vertices[d].pop()
        vertices[d-1].append(j)
        locations[j] = len(vertices[d-1])-1
        degrees[j] -= 1
        #if we delete an edge from a vertex which already has minimal degree
        #then our cursor mind needs to go back one step
        if d==mind: backwards=True
    
    #find the new minimal degree:
    if(backwards):
      mind -= 1
    else:
      while(len(vertices[mind])==0 and mind<nb_nodes-1):
        mind += 1
    
    #compute the new density:
    n -= 1
    if n != 0:
      if density<m/n:
        density=m/n
        subgraph_nodes=n
  
  return density, subgraph_nodes

begin = time.time()
#print(greedy('graph1', 4039, 88234))
#print(greedy('graph2', 54573, 498202))
#print(greedy('graph3', 425957, 1049866))
#print(greedy('graph4', 1696415, 11095298))
#print(greedy('graph5', 4036538, 34681189))
end = time.time()
print("found a result in", end-begin, "seconds")
