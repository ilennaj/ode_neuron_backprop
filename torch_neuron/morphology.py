# Morphology Functions
# morphology.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def Tree_mat(height=2, draw=False):
    '''Returns a torch tensor adjacency matrix of a balanced binary tree'''
    # Make initial binary tree
    g = nx.balanced_tree(r=2,h=height)
    
    # Add soma node which is the last labeled node
    g.add_edge(0,g.number_of_nodes())
    
    # Convert to torch tensor
    tree_mat = torch.Tensor(nx.to_numpy_array(g))
    
    # If draw = True, make a plot of the graph
    if draw:
        nx.draw(g, with_labels=True, font_color='w')
    return(tree_mat)

def Lin_mat(nodes=3, draw=False):
    '''Returns a torch tensor adjacency matrix of a linear path'''
    # Make initial binary tree
    g = nx.path_graph(nodes)
    
    # Convert to torch tensor
    lin_mat = torch.Tensor(nx.to_numpy_array(g))
    
    # If draw = True, make a plot of the graph
    if draw:
        nx.draw(g, with_labels=True, font_color='w')
    return(lin_mat)

# Easy graphing of adjacency matrices
def draw_adj_mat(adj_mat):
    '''Receives torch tensor adjacency matrix and returns a figure'''
    nx.draw(nx.from_numpy_matrix(adj_mat.numpy()), with_labels=True, font_color='w')