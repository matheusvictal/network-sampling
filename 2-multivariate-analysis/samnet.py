import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random as rd
import math
import altair as alt
from scipy.stats import entropy
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# Graph visualizer function for examples
def visualize(G, sample=0, save=0, file_name='example.png', seed=2112):
    if sample == 0:
        pos=nx.spring_layout(G, seed = seed)
        plt.figure(figsize=(15,7))
        nx.draw_networkx(G, node_size=30, with_labels = False ,font_size=16, pos=pos,node_color='crimson',edge_color ='k') 
        if save == 1: 
            plt.savefig(file_name, bbox_inches='tight', dpi = 100)
        plt.show(True)
        
    else:
        pos=nx.spring_layout(G, seed = seed)
        plt.figure(figsize=(15,7))
        nx.draw_networkx(G, node_size=30, with_labels = False ,font_size=16, 
                         pos=pos,node_color='crimson',edge_color ='crimson') 
        nx.draw_networkx(sample, node_size=30, with_labels = False ,font_size=16, 
                         pos=pos,node_color='k',edge_color ='k',width = 2)
        if save == 1: 
            plt.savefig(file_name, bbox_inches='tight', dpi = 100)
        plt.show(True)
    
    return





# Sampling methods implementation

def snowball(G,n,seeds): # G is the graph of interest and n is the sample size and seeds is the number of initial nodes
    import random as rd 
    
    in_sampled_nodes = rd.sample(G.nodes,seeds)
    aux1 = in_sampled_nodes # current stage nodes list
    edgelist = [] # auxiliar list to make sample subgraph later
    
    nodelist = in_sampled_nodes # total nodes samples list
    
    while len(nodelist)<n:
        aux2 = [] # current neighbors list in the current snowball stage

        for i in aux1: # iteration in current nodes
            if len(nodelist)>=n: # verification for the sample size
                break
            neighbors = [] #neighbors list
            
            for k in list(G.neighbors(i)): # adding to the current neighbors list in the snowball stage
                if k not in nodelist:
                    neighbors.append(k)
                else:
                    edgelist.append((i,k))
                    
            aux2 = aux2 + neighbors        
                
            for j in neighbors: # iteration in i's neighbors
                
                #Obs.: we can put a probability p in the command if we want to randomize the neighbor addition to the sample
                edgelist.append((i,j)) # appending to the edgelist
                nodelist.append(j) # count incrementation
                
                if len(nodelist)>=n: # verification for the sample size
                    last_node = j
                    break
                    
        aux1 = aux2 # now, the neighbors are the current stage nodes
        
    sampled_graph = nx.Graph(edgelist)
    
    if len(sampled_graph)>n:
        sampled_graph.remove_node(last_node)
    
    return sampled_graph



# RWS function receives G, n and r (default = 0)
def RWS(G, n, r=0):
    import random as rd

    sampled_node = rd.sample(G.nodes,1) # sampling the node that will start the random walk (returns [node_label])
    
    # Auxiliar variables:
    # sampled_node is a list, not a number, we need a hashable variable for the method .neighbors
    current = sampled_node[0] 
    next_ = 0
    count = 0 # stores the number of sampled nodes
    edgelist = [] # stores sampled edges
    node_list = [] # stores sampled nodes
    node_list.append(current)
    
    # Random walk loop:
    while count<n:
        neighbors = list(G.neighbors(current)) # turn the current node neighbors iterators into a list
        next_ = rd.choice(neighbors) # uniform random choice from neighbors
        edgelist.append((current,next_)) # sampled edge
        node_list.append(next_) # sampled node
        
        # Applying the rewinding probability (rd.random = X ~ U[0,1] , F(x) = x)
        u = rd.random()
        if u < r:
            current = sampled_node[0]
        else:
            current = next_
            
        count = len(Counter(node_list).keys()) # sample size verification
        
    sampled_graph = nx.Graph(edgelist) # builds a nx.Graph object to store the sampled subgraph, mantaining the original labels
    
    return sampled_graph





# IRWS function receives G, n and r (default = 0)
def IRWS(G, n, r=0):
    from collections import Counter
    import random as rd
    
    sampled_node = rd.sample(G.nodes,1) # sampling the node that will start the random walk (returns [node_label])
    
    # Auxiliar variables:
    # sampled_node is a list, not a number, we need a hashable variable for the method .neighbors
    current = sampled_node[0]
    next_ = 0
    count = 0 # stores the number of sampled nodes
    node_list = [] # stores sampled nodes
    node_list.append(current)
    
    # Random walk loop:
    while count<n:
        neighbors = list(G.neighbors(current)) # turn the neighbors iterators into a list
        next_ = rd.choice(neighbors) # uniform random choice from neighbors
        node_list.append(next_)
        
        # Applying the rewinding probability (rd.random = X ~ U[0,1] , F(x) = x)
        u = rd.random() 
        if u < r:
            current = sampled_node[0]
        else:
            current = next_
        
        count = len(Counter(node_list).keys()) # sample size verification
        
    sampled_graph = G.subgraph(node_list) # obtain the induced subgraph of G from the list node_list of vertices
    
    return sampled_graph





def traceroute(G,n): # is the graph of interest and n is the sample size
    
    edgelist = [] # list to store the sampled edges
    count = 0 # counter
    aux = 0 # this auxiliar variable will store the last node to be added in the graph in case the last loop results in a 
    # sample of size n+1
    
    # Principal loop:
    while count < n:
        sample_st = rd.sample(G.nodes,2) # uniformily choses 2 nodes without replacement to act as source and target
        source = sample_st[0]
        target = sample_st[1]
        
        if nx.has_path(G,source,target): # verifies if a path exists between source and target
            path = nx.shortest_path(G, source, target)
            
            for i in range(len(path)-1): # loop to include in the graph the path, node by node
                
                edgelist.append((path[i],path[i+1])) # here 3 things can happen: 0 new nodes are included, 1 new node is 
                # included or 2 new nodes are included, thus, the sample size n can only be passed by 1 in any given case
                
                count = len(nx.Graph(edgelist)) # sample size verification
                if count >= n: # breaks the for loop if the sample size is achieved or passed by 1 (only possible options)
                    aux = path[i+1] # auxiliar variable stores the last node included
                    break
    
    sampled_graph = nx.Graph(edgelist)
    
    if len(sampled_graph)>n: # removes the last added node if sample size is n+1, resulting in a sample size of n
        sampled_graph.remove_node(aux)
        
    # getting the largest connected component of the graph in case the graph obtained is not connected
    if nx.is_connected(sampled_graph) == False:
        largest_cc = max(nx.connected_components(sampled_graph), key=len)
        S = sampled_graph.subgraph(largest_cc).copy() #largest component of graph does not have the desired len
        count = len(S) # the counter assumes the size of S, there are left n - n(S) nodes to sample.
        
        G_nodes = list(G.nodes)
        S_nodes = list(S.nodes)
        
        notin_S_nodes = [item for item in G_nodes if item not in S_nodes] # G nodes that are not in S
        
        while count < n:
            source = rd.sample(S_nodes,1)[0] # sampled node from S to act as source
            target = rd.sample(notin_S_nodes,1)[0] # sampled node from V\S to act as source
            
            if nx.has_path(G,source,target): # verifies if a path exists between source and target
                path = nx.shortest_path(G, source, target)
            
                for i in range(len(path)-1): # loop to include in the graph the path, node by node
                    S.add_edge(path[i],path[i+1]) 
                    count = len(S)
                    if count >= n: 
                        break
    
        sampled_graph = S
    
    return sampled_graph





# MHRW function receives G, n and tol
def MHRW(G, n, tol):
    import random as rd

    sampled_node = rd.sample(G.nodes,1) # sampling the node that will start the random walk (returns [node_label])
    
    # Auxiliar variables:
    # sampled_node is a list, not a number, we need a hashable variable for the method .neighbors
    current = sampled_node[0] 
    next_ = 0
    count = 0 # stores the number of sampled nodes
    edgelist = [] # stores sampled edges
    node_list = [] # stores sampled nodes
    node_list.append(current)
    control = 0
    
    # Random walk loop:
    while count<n:
        neighbors = list(G.neighbors(current)) # turn the current node neighbors iterators into a list
        next_ = rd.choice(neighbors) # uniform random choice from neighbors
        
        kc = G.degree[current]
        kn = G.degree[next_]
        
        u = rd.uniform(0,1)
        
        if u <= min(1,kc/kn):
            edgelist.append((current,next_)) # sampled edge
            node_list.append(next_) # sampled node
            current = next_
            
            count = len(Counter(node_list).keys()) # sample size verification
        
        control = control + 1
        
        if control > tol:
            print("Error")
            break
        
    sampled_graph = nx.Graph(edgelist) # builds a nx.Graph object to store the sampled subgraph, mantaining the original labels
    
    return sampled_graph




# Metrics calculation functions

# sample statistical moments
def moments(l, m):
    M = 0
    N = len(l)
    for i in l:
        M = M + i**m 
    M = M/N 
    return M


# sample shannon entropy
def shannon(vector, bin_s): # function to compute Shannon's Entropy, receives a vector of continuous values and the size of 
    # bins to be considered (vector must be a list)
        
    p_vector = pd.cut(vector, bins = np.arange(0,max(vector)+bin_s,bin_s),include_lowest=True).value_counts() #here,
    # the count of occurence for the specified bin is made
    #print(p_vector)
    p_vector = p_vector.values
    p_vector = p_vector/len(vector) # obtain frequencies of occurence
    #print(p_vector,sum(p_vector))
    H = 0
    
    for p in p_vector:
        if(p > 0):
            H = H - p*math.log(p,10) #acumulates the values of the product of P(k) and log(P(k))
    return H



def total_metrics(G):
    
    # getting the largest connected component of the graph
    if nx.is_connected(G) == False:
        largest_cc = max(nx.connected_components(G), key=len)
        S = G.subgraph(largest_cc).copy()
    else:
        S = G
    
    
    metrics_list = [] # order: assortativity, transitivity, av. shortest path, (first, second, third, fourth moments and 
    # entropy of:) degree, local clustering, betweennes, eigenvector, closeness, communicability, k-core.
    
    # Global metrics
    
    # degree assortativity coefficient
    metrics_list.append(nx.degree_assortativity_coefficient(S))
    
    # transitivity
    metrics_list.append(nx.transitivity(S))
    
    #av. shortest path
    metrics_list.append(nx.average_shortest_path_length(S))
    
    # Complexity coefficient
    
    # <k^2>/<k>
    
    degree_list = dict(S.degree).values()
    complexity_coef = moments(degree_list,2)/moments(degree_list,1)
    
    metrics_list.append(complexity_coef)
    
    
    
    # Local Metrics
    
    # Degree
    
    degree_list = dict(S.degree).values()
    
    for m in range(1,5):
        result = moments(degree_list, m)
        metrics_list.append(result)
        
    metrics_list.append(shannon(list(degree_list),1))    
        
    
    # Local clustering
    
    clustering_list = nx.clustering(S).values()
    
    for m in range(1,5):
        result = moments(clustering_list, m)
        metrics_list.append(result)    
        
    metrics_list.append(shannon(list(clustering_list),0.1))  
        
    
    # Betweenness 
    
    bet_list = nx.betweenness_centrality(S).values()
    
    for m in range(1,5):
        result = moments(bet_list, m)
        metrics_list.append(result) 
        
    metrics_list.append(shannon(list(bet_list),0.1))  
        
    # Closeness
    
    clos_list = nx.closeness_centrality(S).values()
    
    for m in range(1,5):
        result = moments(clos_list, m)
        metrics_list.append(result)
        
    metrics_list.append(shannon(list(clos_list),0.1))  
        
    
    # Communicability
    
    com_list = nx.communicability_betweenness_centrality(S).values()
    
    for m in range(1,5):
        result = moments(com_list, m)
        metrics_list.append(result)
    
    metrics_list.append(shannon(list(com_list),0.1))  
        
        
    # K-core
    
    core_list = nx.core_number(S).values()
    
    for m in range(1,5):
        result = moments(core_list, m)
        metrics_list.append(result)  
        
    metrics_list.append(shannon(list(core_list),1))  
    
    
    #PageRank
    
    pr_list = nx.pagerank(S).values()
    
    for m in range(1,5):
        result = moments(pr_list, m)
        metrics_list.append(result)  
        
    metrics_list.append(shannon(list(pr_list),0.1))  
    
    
    #Eigenvector centrality
    
    eig_list = nx.eigenvector_centrality(S, max_iter = 30000).values()
    
    for m in range(1,5):
        result = moments(eig_list, m)
        metrics_list.append(result)  
        
    metrics_list.append(shannon(list(eig_list),0.1))  
    
        
    # Size of the largest connected component of the graph
    metrics_list.append(len(S))
        
    return metrics_list




def partial_metrics(G):
    
    # getting the largest connected component of the graph
    if nx.is_connected(G) == False:
        largest_cc = max(nx.connected_components(G), key=len)
        S = G.subgraph(largest_cc).copy()
    else:
        S = G
    
    
    metrics_list = [] # order: (first, second, third, fourth moments of:) degree, local clustering, shortest distances
    
    
    # Degree
    
    degree_list = dict(S.degree).values()
    
    for m in range(1,5):
        result = moments(degree_list, m)
        metrics_list.append(result)
        
        
    # Local clustering
    
    clustering_list = nx.clustering(S).values()
    
    for m in range(1,5):
        result = moments(clustering_list, m)
        metrics_list.append(result) 
        
        
        
    # Shortest paths 
    paths = dict(nx.shortest_path_length(S)) # generate a dictionary of dictionaries with the corresponding lengths
    
    path_lengths = [paths[key1][key2] for key1 in paths.keys() for key2 in paths[key1].keys()] # list of lengths
    
    for m in range(1,5):
        result = moments(path_lengths, m)
        metrics_list.append(result) 
        
    
    # Size of the largest connected component of the graph
    metrics_list.append(len(S))
        
    return metrics_list



# Metrics table generators

# graph_list: list of graphs of interest, sizes_list: proportion of size sample in relation to G size, runs: number of samples
# of each type for each network in graph_list for each size in size_list

def total_gen(graph_list, sizes_list, runs):
    
    iterations = len(graph_list)*len(sizes_list)*runs
    
    # list os metrics list for each graph/subgraph
    graph_sample_list = []
    
    # Column names for the dataframe
    metrics_list = [
        "Assortativity",
        "Transitivity",
        "Av. shortest p.",
        "Complexity Coef",
                
        "1m Degree",
        "2m Degree",
        "3m Degree",
        "4m Degree",
        "H Degree",
                
        "1m L. clustering",
        "2m L. clustering",
        "3m L. clustering",
        "4m L. clustering",
        "H L. clustering",
                
        "1m Betweenness",
        "2m Betweenness",
        "3m Betweenness",
        "4m Betweenness",
        "H Betweenness",
                
        "1m Closeness",
        "2m Closeness",
        "3m Closeness",
        "4m Closeness",
        "H Closeness",
                
        "1m Comm",
        "2m Comm",
        "3m Comm",
        "4m Comm",
        "H Comm",
                
        "1m k-core", 
        "2m k-core", 
        "3m k-core", 
        "4m k-core",
        "H k-core",
                
        "1m PageRank", 
        "2m PageRank", 
        "3m PageRank", 
        "4m PageRank",
        "H PageRank",
                
        "1m Eigenvector", 
        "2m Eigenvector", 
        "3m Eigenvector", 
        "4m Eigenvector",
        "H Eigenvector",
                  
        "len",
                    
        "method",
                    
        "net id"
        ] 
    
    iteration_count = 0
 
    for idx, G in enumerate(graph_list):
        
        aux = total_metrics(G)
        aux.append("Original")
        aux.append(idx)
        aux = dict(zip(metrics_list,aux))
        graph_sample_list.append(aux)

        #print("Original", end = " ")
        
        for size in sizes_list:
            
            for _ in range(runs):
                #print(idx, end = " ")
                
                sample = snowball(G, math.floor(size*len(G)),1)
                aux = total_metrics(sample)
                aux.append("SB")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)

                sample = RWS(G, math.floor(size*len(G)))
                aux = total_metrics(sample)
                aux.append("RWS")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                
                sample = IRWS(G, math.floor(size*len(G)))
                aux = total_metrics(sample)
                aux.append("IRWS")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                
                sample = traceroute(G, math.floor(size*len(G)))
                aux = total_metrics(sample)
                aux.append("TR")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                
                sample = MHRW(G, math.floor(size*len(G)), tol = 100000)
                aux = total_metrics(sample)
                aux.append("MHRW")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                           
                iteration_count = iteration_count+1 
                print('Completeness:  ', round(iteration_count/iterations,3))
            



    df = pd.DataFrame(graph_sample_list)    
        
    return df 






# graph_list: list of graphs of interest, sizes_list: proportion of size sample in relation to G size, runs: number of samples
# of each type for each network in graph_list for each size in size_list

def partial_gen(graph_list, sizes_list, runs):
    
    iterations = len(graph_list)*runs
                      
    # list os metrics list for each graph/subgraph
    graph_sample_list = []
    
    # Column names for the dataframe
    metrics_list = [
        "1m Degree",
        "2m Degree",
        "3m Degree",
        "4m Degree",
                
        "1m L clustering",
        "2m L clustering",
        "3m L clustering",
        "4m L clustering",
                
        "1m s path l.",
        "2m s path l.",
        "3m s path l.",
        "4m s path l.",
                  
        "len",
                    
        "method",
                    
        "net id"
                ]  
    
    iteration_count = 0
                      
    for idx, G in enumerate(graph_list):
        
        aux = partial_metrics(G)
        aux.append("Original")
        aux.append(idx)
        aux = dict(zip(metrics_list,aux))
        graph_sample_list.append(aux)

        for size in sizes_list:
            
            for _ in range(runs):
                # print(idx, end = " ")
                sample = snowball(G, math.floor(size*len(G)),1)
                aux = partial_metrics(sample)
                aux.append("SB")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)

                sample = RWS(G, math.floor(size*len(G)))
                aux = partial_metrics(sample)
                aux.append("RWS")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                
                sample = IRWS(G, math.floor(size*len(G)))
                aux = partial_metrics(sample)
                aux.append("IRWS")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                
                sample = traceroute(G, math.floor(size*len(G)))
                aux = partial_metrics(sample)
                aux.append("TR")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                
                sample = MHRW(G, math.floor(size*len(G)), tol = 100000)
                aux = partial_metrics(sample)
                aux.append("MHRW")
                aux.append(idx)
                aux = dict(zip(metrics_list,aux))
                graph_sample_list.append(aux)
                      
                iteration_count = iteration_count+1
                print('Completeness:  ', round(iteration_count/iterations,3))

    df = pd.DataFrame(graph_sample_list)    
        
    return df 