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
        
                print("***Cloest neighbor test, before the application of the syntactic extractor***")
                self.closest_vector_test(apply_transformation = False)
                print("***Cloest neighbor test, after the application of the syntactic extractor***")
                self.closest_vector_test(apply_transformation = True, verbose = False)
                
                print("***Vector-dep association test, before the application of the syntactic extractor***")
                self.vector_dep_association_test(apply_transformation = False, verbose = False)
                print("***Vector-dep association test, After the application of the syntactic extractor***")
                self.vector_dep_association_test(apply_transformation = True, verbose = True)
                
        def closest_vector_test(self, n = 1000, verbose = False, apply_transformation = False):
        
                print("Performing closest vector test...")
                
                vec_lists = self.vec_lists
                all_vecs = [copy.deepcopy(v) for sent_vecs in vec_lists for v in sent_vecs] # all vecs in a single list.
                
                if apply_transformation:
                
                        for i,v in enumerate(all_vecs):
                        
                               v.vec = self.extractor.extract(v.vec.copy().reshape(1, -1)).reshape(-1)
                
                random.shuffle(all_vecs)
                
                good, bad = 0., 0.
                
                
                for i in range(n):
                
                        vec = all_vecs[i]
                        nearest_neighbor = Vector.get_closest_vector(vec, all_vecs)
                        
                        if vec.dep == nearest_neighbor.dep:
                        
                                good += 1
                        
                        else:
                                bad += 1
                        
                        if verbose:
                        
                                print("key: {} \n value: {}\n key-dep: {} \n value-dep: {} \n ------------------------------------------------".format(vec, nearest_neighbor, vec.dep, nearest_neighbor.dep))
                        
                acc = good / (good + bad)
                print("Average accuracy is {}".format(acc))
        
        def vector_dep_association_test(self, n = 1000, clustering = "kmeans", num_clusters = 50, verbose = False, apply_transformation = False):
        
                print("Perfoming vector-dependency association test")
                print("Performing clustering...")
                
                vec_lists = self.vec_lists
                all_vecs = [copy.deepcopy(v.get_vector()) for sent_vecs in vec_lists for v in sent_vecs] # all vecs in a single list.
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
                
                
        def _load_sents(self, fname = "sents_f", max_length = 25) -> List[List[str]]:
        
                print("Loading sentences...")
                
                with open(fname, "r") as f:
                        lines = f.readlines()
                        lines =  [line.strip().split(" ") for line in lines]
                        
                if max_length is not None:
                        lines = list(filter(lambda sentence: len(sentence) < max_length, lines))
                
                lines = lines[:]
                
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
