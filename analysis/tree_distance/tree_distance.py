import spacy
from benepar.spacy_plugin import BeneparComponent
from nltk.tree import Tree
import kernel
import numpy as np
import utils


def get_tree_size(tree):

        return len(tree.treepositions())
        
def get_tree(sentence, remove_leaves = True):

        nlp = spacy.load('en')
        nlp.add_pipe(BeneparComponent('benepar_en'))
        doc = nlp(sentence)
        sent = list(doc.sents)[0]
        tree_str =sent._.parse_string
        tree = Tree.fromstring(tree_str)
        
        if remove_leaves:
        
                for pos in tree.treepositions('leaves'):
                
                        tree[pos] = 'w'
        return tree

def get_kernel(sent1, sent2, normalize = True, remove_leaves = True):

        t1 = get_tree(sent1, remove_leaves = remove_leaves)
        t2 = get_tree(sent2, remove_leaves = remove_leaves)
        #print(get_tree_size(t1))
        
        print(t1.pretty_print())
        print("---------------------------------------")
        print(t2.pretty_print())
        
        K = kernel.Kernel(alpha = 0.5)
               
        k = K(t1,t2)
        
        if normalize:
        
                k /= np.sqrt(K(t1,t1) * K(t2,t2))
        
        return k

def get_tree_edit_distance(sent1, sent2):

        get_label_func = kernel.label
        get_children_func = kernel.children
        get_dist_func = lambda node1, node2: 0 if get_label_func(node1) == get_label_func(node2) else 1
        
        import zss
        t1 = get_tree(sent1, remove_leaves = True)
        t2 = get_tree(sent2, remove_leaves = True)
        dist = zss.simple_distance(t1, t2, get_children_func, get_label_func, get_dist_func)
        
        return dist/(np.sqrt(get_tree_size(t1) * get_tree_size(t2)))
        
sent1 = "if there is a consensus for keeping all articles on universities and colleges and comparable institutions , i think that ought to be made clear ."
sent2 = "gng is neither necessary nor sufficient to create an article , however common it is to be found in articles and afd discussions ."

#sent1 = "The man sees the mouse"
#sent2 = "a tree watches a flower"

print(get_kernel(sent1, sent2))
#print(get_kernel(sent1, sent3))
print(get_tree_edit_distance(sent1, sent2))
#print(get_tree_edit_distance(sent1, sent3))
