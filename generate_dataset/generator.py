from generator_base import EquivalentSentencesGenerator
import spacy
import utils
from typing import DefaultDict, List
from collections import defaultdict
import random
import os.path
import tqdm
import pickle

class POSBasedEGenerator(EquivalentSentencesGenerator):

	def __init__(self, data_filename, pos_tags_to_replace, num_sentences):

		super().__init__(data_filename, num_sentences)
		
		self.nlp = spacy.load('en_core_web_sm')
		self.data_filename = data_filename
		self.pos2words = self._get_POS2words_mapping()
		self.pos_tags_to_replace = pos_tags_to_replace

	def _get_POS2words_mapping(self) -> DefaultDict[str, set]:

		"""
		Iterate over the dataset, and find the words belonging to each POS tag.
		return: pos2words, a dictionary mapping pos tags (strings) to sets of words.
		"""
		
		pos2words_filename = utils.DEFAULT_PARAMS["pos2words_filename"]
		
		if os.path.isfile(pos2words_filename):
			
			pickleFile = open(pos2words_filename, 'rb')
			pos2words = pickle.load(pickleFile)
		
		else:
		
			print("Collecting POS:words mapping...")
			
			pos2words = defaultdict(set)

			sentences = utils.read_sentences(self.data_filename)
			
			for sentence in tqdm.tqdm(sentences):

				pos_tags = self._get_pos_tags(sentence)

				for (w, pos_tag) in zip(sentence, pos_tags):

					pos2words[pos_tag].add(w)
					
			pickleFile = open(pos2words_filename, 'wb')
			pickle.dump(pos2words, pickleFile)
			
		return pos2words

	def _get_pos_tags(self, sentence: List[str]) -> List[str]:
		
		doc = self.nlp(" ".join(sentence))
		pos_tags = []

		for token in doc:

			pos_tags.append(token.pos)

		return pos_tags	


	def get_equivalent_sentences(self, original_sentence: List[str],) -> List[List[str]]:

		pos_tags = self._get_pos_tags(original_sentence)
		equivalent_sentences = []

		for i in range(self.num_sentences):

			sentence = []

			for j, (w, pos_tag) in enumerate(zip(original_sentence, pos_tags)):

				if pos_tag in self.pos_tags_to_replace:

					sentence.append(random.choice(self.pos2words[pos_tag]))
				else:
					sentence.append(w)

			equivalent_sentences.append(sentence)

		return equivalent_sentences

	
			




