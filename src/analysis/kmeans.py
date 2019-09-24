

import pickle
from collections import defaultdict
import sys
sys.path.append('../src/generate_dataset')
import typing
from typing import List, Tuple, Dict
import evaluate3 as evaluate
import numpy as np
import spacy
import nltk
import syntactic_extractor
from collections import defaultdict
import sklearn
from sklearn import cluster
import random

Sentence_vector = typing.NamedTuple("Sentence_vector",
                                    [('sent_vectors', np.ndarray), ('sent_str', List[str]),
                                     ("doc", spacy.tokens.Doc), ("tree", nltk.Tree)])
Word_vector = typing.NamedTuple("Word_vector", [('word_vector', np.ndarray), ('sentence', List[str]),
                                                ("doc", spacy.tokens.Doc), ("index", int), ("tree", nltk.Tree)])
                                                
                                                
def get_sentence_objects(path="sent_objs.pickle"):
        
        print("Loading sents...")
        
        with open(path, "rb") as f:
                sentence_reprs = pickle.load(f)
        
        print("Done.")
        return sentence_reprs

def load_extractor():
        
        return syntactic_extractor.TripletLossModelExtractor()

def get_words(sentence_reprs, extractor):

        print("Loading words...")
        words_reprs = evaluate.sentences2words(sentence_reprs, num_words = 250000, ignore_function_words = False)
        vecs = [w.word_vector for w in words_reprs]
        deps = [w.doc[w.index].dep_ for w in words_reprs]
        sents = [w.sentence for w in words_reprs]
        indices = [w.index for w in words_reprs]
        
        if extractor is not None:
                for i,v in enumerate(vecs):
                        vecs[i] = extractor.extract(v).reshape(-1)
                
        words = [(v, dep, sent, index) for (v,dep,sent,index) in zip(vecs, deps, sents, indices)]
        return words


def get_dep2words_mapping(words):

        dep2words = defaultdict(list)
        for w in words:
        
                v, dep, sent, index = w
                dep2words[dep].append((v, sent, index))
        return dep2words
        
def perform_kmeans(dep2words, num_clusts = 10):

        dep2clust2words = dict()
        
        for dep_edge, words in dep2words.items():
                
                
                if len(words) < 50: continue
                
                print("Performing clustering for dep edge = {}".format(dep_edge))
                
                vecs = [v for (v, sent, index) in words]

                kmeans = sklearn.cluster.KMeans(n_clusters = num_clusts)
                kmeans.fit(vecs)
                labels = kmeans.labels_
                label2words = defaultdict(list)
                
                for (clust, w) in zip(labels,words):
                
                       label2words[clust].append(w)
                
                dep2clust2words[dep_edge] =  label2words
                        
        return dep2clust2words
                

def get_sent_string(sent: List[str], index, color = "blue"):

        before, w, after = sent[:index], sent[index], sent[index + 1:]
        before = " ".join(before)
        after = " ".join(after)
        w = '<b><font color="{}">'.format(color) + w + '</font></b>'
        return '<font size = "5">' + before + " " + w + " " + after + "</font>"

def create_html(dep2clust2words, examples_per_cluster = 50):

        html = ""
        
        for dep_edge, clust2words in dep2clust2words.items():
        
                html += "<h1>"+ "Dep Edge = {}".format(dep_edge) + "</h1>\n"
                clusters_and_words = list(clust2words.items())
                clusters_and_words = sorted(clusters_and_words, key = lambda tup: int(tup[0]))
                
                for clust, words in clusters_and_words:
                
                        random.shuffle(words)
                        
                        html += "<h2>"+ "Cluster = {}".format(clust) + "</h2>\n"
                        html += "<ul>\n" 
                        
                        for (v, sent, index) in words[:examples_per_cluster]:
                        
                                sent_str = get_sent_string(sent, index, "blue")
                                html += "<li>" + sent_str + "</li>\n"
                        
                        html+="</ul>\n"               
                        
        with open("clusters-after.html", "w", encoding = "utf-8") as f:
          
                f.write(html)           
        
def main():

        sents = get_sentence_objects()
        extractor =  load_extractor()
        words = get_words(sents, extractor=extractor)     
        dep2words = get_dep2words_mapping(words)
        dep2clust2words = perform_kmeans(dep2words, num_clusts = 10)
        create_html(dep2clust2words)



main()     
                
                
