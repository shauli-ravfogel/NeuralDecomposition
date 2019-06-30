from syntactic_extractor import SyntacticExtractor
from typing import List
import copy
import numpy as np

class Vector(object):
    
    def __init__(self, vec, sentence, index):
        
        self.vec = vec
        self.sentence = sentence
        self.index = index
        self.size = np.linalg.norm(self.vec)
    
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
		
	def _load_elmo(self):
	
		options_file = "../data/external/elmo_2x4096_512_2048cnn_2xhighway_options.json"
		weight_file = "../data/external/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		return FastElmo(options_file, weight_file)
	
	def _run_elmo(sentences: List[List[str]]) -> List[np.ndarray]:
	
	
	        elmo_embeddings = self.elmo.embed_sentences(sentences)
	        all_embeddings = []
	        
	        for sent_emn in elmo_embeddings:
	        
	                last_layer = sent_emn[:, -1]
	                all_embeddings.append(last_layer)
	        
	        return all_embeddings
	        
		
	def _load_sents(self, fname = "sents_f", max_length = 25) -> List[List[str]]:
	
		with open(fname, "r") as f:
			lines = f.readlines()
			lines =  [line.strip().split(" ") for line in lines]
			
		if max_length is not None:
			lines = list(filter(lambda sentence: len(sentence) < max_length, lines))
		return lines
		
	def _parse(self, sentences: List[List[str]]) -> List[List[str]]:
	                                             
		tokens_dict = {" ".join(sentence):sentence for sentence in sentences}
		def custom_tokenizer(text): return tokens_dict[text]
		nlp = spacy.load('en_core_web_sm')
		#parser = nlp.create_pipe("parser")
		all_deps = []
		count = 0
		 
		for sent in sentences:
		
			doc = spacy.tokens.Doc(vocab=nlp.vocab, words=sent)
			for name, proc in nlp.pipeline:
				doc = proc(doc)
				
			deps = [token.dep_ for token in doc]
			all_deps.append(deps)
			
			assert len(deps) == len(sent)
		
		return all_deps

	def _list_vectors(self, sents_embeddings: List[np.ndarray], sents: List[List[str]]) -> List[List[Vector]]:
	
	
		"""
		Transform the list of all state vectors (as numpy arrays) to a list of Vector objects.
		
		Parameters
		
		---------
		sents_embeddings: ``List[np.ndarray]``, required
		
		        A list of ELMO embeddings for all sentences. Each list element is ELMO embedding 
		        of a different sentence, with dimensions (sentence_length, 1024)
		
		sents: `` List[List[str]]``, required
		
		        A list of all sentences. Each list contains a list representing a different sentence.
		        
		Returns
		---------
		
		all_vectors: ``List[List[Vector]``
		       
		       A list of lists of Vector objects. all_vectors[i][j] is the representation of the jth word
		       in the ith sentence.
		        
		"""
		
		print("Creating Vector objects required for nearest neighbor search...")
		
		assert len(sents) == len(sents_embeddings)
		
		num_sentences = len(sents)
		sents_indices_and_vecs =  zip(range(num_sentences), sents_embeddings)
		all_vectors = []
		
		for sent_index, sent_vectors in sents_indices_and_vecs:
		
		        sent_vectors = []
		        
		        assert len(sents[sent_index]) == sent_vectors.shape[0]
		        
			for i, (w, vec) in enumerate(zip(sents[sent_index], sent_vectors)):   
			
				v = Vector(vec.detach().numpy(), sents[sent_index], i)
				sent_vectors.append(v)
				
			all_vectors.append(sent_vectors)
				
		return all_vectors

e = Evaluator()
