import spacy
from benepar.spacy_plugin import BeneparComponent
from nltk.tree import Tree
import kernel
import zss
import numpy as np
import utils
from tqdm.auto import tqdm
import typing

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en'))


def _get_tree_size(tree):
    return len(tree.treepositions())


def get_similarity_scores(sents_lst_1, sents_lst_2):
    kernel_similarities = []

    print("Creating trees...")
    trees_lst = [(_create_tree(s1), _create_tree(s2)) for (s1, s2) in
                 tqdm(zip(sents_lst_1, sents_lst_2), total=len(sents_lst_1))]
    print("Calculating kernel similarities...")
    kernel_similarities = [_kernel_similarity(t1, t2) for (t1, t2) in tqdm(trees_lst, total=len(trees_lst))]
    print("Calculating edit distance similarities...")
    edit_similarities = [_edit_distance_similarity(t1, t2) for (t1, t2) in tqdm(trees_lst, total=len(trees_lst))]

    return (kernel_similarities, edit_similarities)


def _create_tree(sentence, remove_leaves=True):
    sentence = " ".join(sentence)

    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    tree_str = sent._.parse_string
    tree = Tree.fromstring(tree_str)

    if remove_leaves:

        for pos in tree.treepositions('leaves'):
            tree[pos] = 'w'
    return tree


def _kernel_similarity(t1, t2, normalize=True, remove_leaves=True):
    K = kernel.Kernel(alpha=0.75)
    k = K(t1, t2)

    if normalize:
        k /= np.sqrt(K(t1, t1) * K(t2, t2))

    return k


def _edit_distance_similarity(t1, t2):
    get_label_func = kernel.label
    get_children_func = kernel.children
    get_dist_func = lambda node1, node2: 0 if get_label_func(node1) == get_label_func(node2) else 1

    dist = zss.simple_distance(t1, t2, get_children_func, get_label_func, get_dist_func)

    return 1. - dist / (np.sqrt(_get_tree_size(t1) * _get_tree_size(t2)))


"""      
sent1 = "the 2003 festival tickets had a code on them , which would allow festival goers to download tracks from bands which had played ."
sent2 = "the season premiere for the 2001 season featured a sketch that was considered offensive by conservatives who threatened a boycott ."

#sent1 = "The man sees the mouse"
#sent2 = "a tree watches a flower"

print(get_kernel(sent1, sent2))
#print(get_kernel(sent1, sent3))
print(get_tree_edit_distance(sent1, sent2))
#print(get_tree_edit_distance(sent1, sent3))
"""
