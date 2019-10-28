import spacy
import nltk
from nltk.tree import Tree
import kernel
import zss
import numpy as np
import utils
from tqdm.auto import tqdm
import typing
from typing import List
import copy

def _get_tree_size(tree: nltk.tree.Tree):

        return len(tree.treepositions())
        
def get_similarity_scores(sents_lst_1, sents_lst_2):

        kernel_similarities = []
        
        print("Creating trees...")
        trees_lst = [(s1.tree, s2.tree) for (s1,s2) in zip(sents_lst_1,sents_lst_2)]
        print("Calculating kernel similarities...")
        kernel_similarities = [_kernel_similarity(copy.deepcopy(t1),copy.deepcopy(t2)) for (t1,t2) in tqdm(trees_lst, total = len(trees_lst))]
        return kernel_similarities
        
        #print("Calculating edit distance similarities...")
        #edit_similarities = [_edit_distance_similarity(t1, t2) for (t1, t2) in tqdm(trees_lst, total = len(trees_lst))]
        #return edit_similarities
        #return (kernel_similarities, edit_similarities)



def _kernel_similarity(t1, t2, normalize = False, remove_leaves = True):

        if remove_leaves:
	
	        for pos in t1.treepositions('leaves'):
	            t1[pos] = 'w'
	        for pos in t2.treepositions('leaves'):
	            t2[pos] = 'w'
	            	            
        K = kernel.Kernel(alpha = 1)
        k = K(t1,t2)
        
        if normalize:
        
                k /= np.sqrt(K(t1,t1) * K(t2,t2))
        
        return k

def _edit_distance_similarity(t1, t2):

        get_label_func = kernel.label
        get_children_func = kernel.children
        get_dist_func = lambda node1, node2: 0 if get_label_func(node1) == get_label_func(node2) else 1
        
        dist = zss.simple_distance(t1, t2, get_children_func, get_label_func, get_dist_func)
        
        return 1. - dist/(_get_tree_size(t1) + _get_tree_size(t2))

