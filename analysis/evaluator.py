from syntactic_extractor import SyntacticExtractor
from allennlp.commands.elmo import ElmoEmbedder
from typing import List
import copy
import numpy as np
import random
random.seed(0)
from collections import Counter, defaultdict
import spacy
import tqdm
from sklearn.cluster import KMeans
import copy
import termcolor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('tkagg')
import sys
sys.path.append('../src/generate_dataset')
import utils
import pylab
                
                
class Vector(object):
    
    def __init__(self, vec, sentence, index, dep):
        
        self.vec = vec
        self.sentence = sentence
        self.index = index
        self.size = np.linalg.norm(self.vec)
        self.dep = dep
    
    def get_word(self):
        
        return self.sentence[self.index]
    
    def get_vector(self): 
        
        return self.vec
    
    def get_sentence(self):
        
        return self.sentence
    
    def get_index(self):
        
        return self.index
    
    def get_size(self):
        
        return self.size
    
    def __str__(self):
        
        words = self.get_sentence()
        i = self.get_index()
        before = " ".join(words[:i])
        after = " ".join(words[i + 1:])
        word = "***"+termcolor.colored(self.get_word(), "blue", attrs = ['bold'])+"***"
        sent = '""' + before + " " + word + " " + after + '"' + "***WORD: {} ***".format(self.get_word())
        return sent
    
    def similarity(self, other):
        
        if other is self: return -np.inf
        
        return self.get_vector().dot(other.get_vector())/(self.get_size() * other.get_size())
    
    @staticmethod
    def get_closest_vector(vec, vecs):
    
        closest = max(vecs, key = lambda vector: vector.similarity(vec))
        return closest
        








        
class Evaluator(object):

        def __init__(self, extractor: SyntacticExtractor):
        
                self.extractor = extractor
                self.sentences = self._load_sents()
                self.parsed_sentences = self._parse(self.sentences)
                self.elmo = self._load_elmo()
                self.elmo_embeddings = self._run_elmo(self.sentences)
                self.vec_lists = self._list_vectors(self.elmo_embeddings, self.sentences, self.parsed_sentences)
        
        
        def test(self):
        
                #self.plot()

                self.get_closest_sentence()

                print("***Cloest neighbor test, before the application of the syntactic extractor***")
                self.closest_vector_test(apply_transformation = False, verbose = True)
                print("\n\n\n\n\n\n\n\n\n\n")
                print("***Cloest neighbor test, after the application of the syntactic extractor***")
                self.closest_vector_test(apply_transformation = True, verbose = True)
                
                
                #print("***Vector-dep association test, before the application of the syntactic extractor***")
                #self.vector_dep_association_test(apply_transformation = False, verbose = False)
                #print("***Vector-dep association test, After the application of the syntactic extractor***")
                #self.vector_dep_association_test(apply_transformation = True, verbose = False)
                

        def plot(self):
        
        
                random.seed(0)

                vec_lists = self.vec_lists
                all_vecs = [v.get_vector().copy() for sent_vecs in vec_lists for v in sent_vecs if v.get_word() not in utils.DEFAULT_PARAMS["function_words"]] # all vecs in a single list.
                #all_vecs = [v for v in all_vecs if v.get_word() not in utils.DEFAULT_PARAMS["function_words"]]

                all_deps = [v.dep for sent_vecs in vec_lists for v in sent_vecs if v.get_word() not in utils.DEFAULT_PARAMS["function_words"]]
                deps_set = set(all_deps)
                dep2int = {d:i for i,d in enumerate(deps_set)}
                deps = [dep2int[dep] for dep in all_deps]


                for i,v in enumerate(all_vecs):
                        
                   all_vecs[i] = self.extractor.extract(v.copy().reshape(1, -1)).reshape(-1)
                
                k = 3500
                
                all_vecs = np.array(all_vecs[:k])
                deps = deps[:k]                
                print ("calculating projection...")
                
                proj = TSNE(n_components=2).fit_transform(all_vecs)
                
                
                fig, ax = plt.subplots()
                
                xs, ys = proj[:,0], proj[:,1]
                xs, ys = list(xs), list(ys)
                

                #ax.scatter(xs, ys, color = deps, cmap=pylab.cm.cool)
                pylab.scatter(xs, ys, c=deps, cmap=pylab.cm.cool)
                pylab.show()
                #plt.show()                
                #plt.show()
        
        def get_closest_sentence(self, n = 1000, apply_transformation  = False):
        
                random.seed(0)
                vec_lists = self.vec_lists
                
                sents_repres = []
                sents_strs = []
                
                for i, sent_vecs in tqdm.tqdm(enumerate(self.vec_lists)):
                
                        #sent_vecs_np = np.array([self.extractor.extract([v.get_vector()]).reshape(-1)[1:] for v in sent_vecs])
                        sent_vecs_np = np.array([self.extractor.extract([v.get_vector()]).reshape(-1)[:] for v in sent_vecs])
                        #sent_vecs_np = np.stack([v.get_vector()[1:] for v in sent_vecs])
                        sent_lst = sent_vecs[0].sentence#.split(" ")
                        good_indices = np.array([i for i in range(len(sent_lst)) if sent_lst[i] not in utils.DEFAULT_PARAMS["function_words"]])
                        sent_vecs_np = sent_vecs_np[good_indices]
                        sent_mean_vec = np.mean(sent_vecs_np, axis = 0)
                        sents_repres.append(sent_mean_vec)
                        sents_strs.append(sent_vecs[0].get_sentence())
                
                sents_repres = np.array(sents_repres)
                for i in range(len(sents_repres)):
                
                        sents_repres[i] /= np.linalg.norm(sents_repres[i])
                        
                print(sents_repres[0].shape)
                
                for i in range(min(n, len(self.vec_lists))):
                
                        similarity_scores = sents_repres.dot(sents_repres[i].T)
                        three_largest = similarity_scores.argsort()[-3:][::-1]
                        j = three_largest[1]
                        sent_i, sent_j = sents_strs[i], sents_strs[j]
                        
                        print(" ".join(sent_i))
                        print("-------------------")
                        print(" ".join(sent_j))
                        print("====================================================")
                        
                                
                
                        
                                                        
        def closest_vector_test(self, n = 1000, verbose = False, apply_transformation = False):
        
                random.seed(0)
                
                print("Performing closest vector test...")
                
                vec_lists = self.vec_lists
                all_vecs = [copy.deepcopy(v) for sent_vecs in vec_lists for v in sent_vecs] # all vecs in a single list.
                all_vecs = [v for v in all_vecs if v.get_word() not in utils.DEFAULT_PARAMS["function_words"]]
                                
                if apply_transformation:
                
                        for i,v in enumerate(all_vecs):
                        
                               v.vec = self.extractor.extract([v.vec.copy()]).reshape(-1)
                
                random.shuffle(all_vecs)
                
                good, bad = 0., 0.
                
                
                for i in range(min(len(all_vecs), 1000)):
                
                        vec = all_vecs[i]
                        nearest_neighbor = Vector.get_closest_vector(vec, all_vecs)
                        correct = False
                        
                        if vec.dep == nearest_neighbor.dep:
                                correct = True
                                good += 1
                        
                        else:
                                bad += 1
                        
                        if verbose and i < 300:
                        
                                print("key: {} \n value: {}\n key-dep: {} \n value-dep: {} \n ------------------------------------------------".format(vec, nearest_neighbor, vec.dep, nearest_neighbor.dep))
                        
                acc = good / (good + bad)
                print("Average accuracy is {}".format(acc))
        
        def vector_dep_association_test(self, n = 1000, clustering = "kmeans", num_clusters = 50, verbose = False, apply_transformation = False):
        
                random.seed(0)
                print("Perfoming vector-dependency association test")
                print("Performing clustering...")
                
                vec_lists = self.vec_lists
                all_vecs = [v.get_vector().copy() for sent_vecs in vec_lists for v in sent_vecs if v.get_word() not in utils.DEFAULT_PARAMS["function_words"]] # all vecs in a single list.
                #all_vecs = [v for v in all_vecs if v.get_word() not in utils.DEFAULT_PARAMS["function_words"]]
                
                if apply_transformation:
                
                        for i,v in enumerate(all_vecs):
                        
                                all_vecs[i] = self.extractor.extract(v.copy().reshape(1, -1)).reshape(-1)
                        
                all_deps = [dep for sent_deps in self.parsed_sentences for dep in sent_deps]
                
                assert len(all_vecs) == len(all_deps)
                
                random.shuffle(all_vecs)
                
                if clustering == "kmeans":
                
                        clustering = KMeans(n_clusters=num_clusters, random_state=0).fit(all_vecs)
                
                print("Counting coocurrences...")
                      
                labels = clustering.labels_
                clusts_and_deps = zip(labels, all_deps)
                dep_clust_coocurrences = Counter(clusts_and_deps)
                
                # Create clust:dep count mapping
                
                clust2dep = defaultdict(dict)
                for (clust, dep), count in dep_clust_coocurrences.items():
                
                     clust2dep[clust][dep] = count

                # Noramlize counts
                
                for clust, dep_count_dict in clust2dep.items():
                
                        sum_counts = sum(dep_count_dict.values())
                        for k,v in dep_count_dict.items():
                        
                                dep_count_dict[k] /= (1.*sum_counts)

                # Calculate Entropy
                
                mean_ent = 0.
                
                for clust, dep_count_dict in clust2dep.items():
                
                                deps, probs = list(dep_count_dict.keys()), list(dep_count_dict.values())
            
                                entropy = np.sum([-prob * np.log(prob) if prob > 1e-5 else 0 for prob in probs])
                                mean_ent += entropy 
                                
                                if verbose:
                                        
                                        max_ind = np.argmax(probs)
                                        max_dep = deps[max_ind]
                                        prob = probs[max_ind]
                                        
                                        print("Cluster {} is most associcated with dep edge {}; coocurrence prob: {}".format(clust, max_dep, prob))
             
                mean_ent /= len(clust2dep.keys())
                print("Average entropy is {}".format(mean_ent))               
                                            
        def _load_elmo(self):
        
                print("Loading ELMO...")
                
                options_file = "../data/external/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                weight_file = "../data/external/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                return ElmoEmbedder(options_file, weight_file, cuda_device=0)
                return ElmoEmbedder(options_file, weight_file)
        
        def _run_elmo(self, sentences: List[List[str]]) -> List[np.ndarray]:
        
                print("Running ELMO...")
                
                elmo_embeddings = []# self.elmo.embed_sentences(sentences)
                for sent in tqdm.tqdm(sentences):
                
                        elmo_embeddings.append(self.elmo.embed_sentence(sent))
                        
                all_embeddings = []
                
                for sent_emn in elmo_embeddings:
                
                        last_layer = sent_emn[-1, :, :]
                        all_embeddings.append(last_layer)
                
                return all_embeddings
                
                
        def _load_sents(self, fname = "sents_f", max_length = 35) -> List[List[str]]:
        
                print("Loading sentences...")
                
                with open(fname, "r") as f:
                        lines = f.readlines()
                        lines =  [line.strip().split(" ") for line in lines]
                        
                if max_length is not None:
                        lines = list(filter(lambda sentence: len(sentence) < max_length, lines))
                
                lines = lines[:45000]
                
                return lines
                
        def _parse(self, sentences: List[List[str]]) -> List[List[str]]:
                             
                print("Parsing...")
                                        
                tokens_dict = {" ".join(sentence):sentence for sentence in sentences}
                def custom_tokenizer(text): return tokens_dict[text]
                nlp = spacy.load('en_core_web_sm')
                #parser = nlp.create_pipe("parser")
                all_deps = []
                count = 0
                 
                for sent in tqdm.tqdm(sentences):
                
                        doc = spacy.tokens.Doc(vocab=nlp.vocab, words=sent)
                        for name, proc in nlp.pipeline:
                                doc = proc(doc)
                                
                        deps = [token.dep_ for token in doc]
                        all_deps.append(deps)
                        
                        assert len(deps) == len(sent)
                
                return all_deps

        def _list_vectors(self, sents_embeddings: List[np.ndarray], sents: List[List[str]], parsed_sents:  List[List[str]]) -> List[List[Vector]]:
        
        
                """
                Transform the list of all state vectors (as numpy arrays) to a list of Vector objects.
                
                Parameters
                
                ---------
                sents_embeddings: ``List[np.ndarray]``, required
                
                        A list of ELMO embeddings for all sentences. Each list element is ELMO embedding 
                        of a different sentence, with dimensions (sentence_length, 1024)
                
                sents: `` List[List[str]]``, required
                
                        A list of all sentences. Each list contains a list representing a different sentence.
                
                parsed_sents: `` List[List[str]]``, required
                        
                        A list of all dependency edges. Each list contains a list representing a different sentence.
                        parsed_sents[i][j] contains the dependency edge between the jth word in the ith sentence,
                        and its parent.
                        
                Returns
                ---------
                
                all_vectors: ``List[List[Vector]``
                       
                       A list of lists of Vector objects. all_vectors[i][j] is the representation of the jth word
                       in the ith sentence.
                        
                """
                
                print("Creating Vector objects required for nearest neighbor search...")
                
                assert len(sents) == len(sents_embeddings) == len(parsed_sents)
                
                num_sentences = len(sents)
                sents_indices_and_vecs =  zip(range(num_sentences), sents_embeddings)
                all_vectors = []
                
                for sent_index, sent_vectors in tqdm.tqdm(sents_indices_and_vecs):
               
                        sent_vectors_lst = []
                        
                        assert len(sents[sent_index]) == sent_vectors.shape[0]
                        
                        for i, (w, dep, vec) in enumerate(zip(sents[sent_index], parsed_sents[sent_index], sent_vectors)):   
                        
                                v = Vector(vec, sents[sent_index], i, dep)
                                sent_vectors_lst.append(v)
                                
                        all_vectors.append(sent_vectors_lst)
                                
                return all_vectors

#e = Evaluator()
