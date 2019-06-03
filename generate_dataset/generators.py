from generator_base import EquivalentSentencesGenerator
import spacy
import utils
from typing import DefaultDict, List
from collections import defaultdict, Counter
import random
import os.path
import tqdm
import pickle
import gensim
from scipy import spatial
# an approximate knn: https://github.com/yahoojapan/NGT
import ngtpy  # pip install ngtpy


class POSBasedEGenerator(EquivalentSentencesGenerator):

    def __init__(self, data_filename, pos_tags_to_replace, num_sentences):

        super().__init__(data_filename, num_sentences)

        self.nlp = spacy.load('en_core_web_sm')
        self.data_filename = data_filename
        self.pos2words = self._get_POS2words_mapping()
        self.pos_tags_to_replace = pos_tags_to_replace

    def _get_POS2words_mapping(self, min_occurrence=50) -> DefaultDict[str, set]:

        """
		Iterate over the dataset, and find the words belonging to each POS tag.
		return: pos2words, a dictionary mapping pos tags (strings) to sets of words.
		"""

        pos2words_filename = utils.DEFAULT_PARAMS["pos2words_filename"]

        if os.path.isfile(pos2words_filename):

            with open(pos2words_filename, 'rb') as f:
                pos2words = pickle.load(f)

        else:

            print("Collecting POS:words mapping...")

            pos2words = defaultdict(list)

            sentences = utils.read_sentences(self.data_filename)

            for sentence in tqdm.tqdm(sentences):

                pos_tags = self._get_pos_tags(sentence)

                for (w, pos_tag) in zip(sentence, pos_tags):
                    pos2words[pos_tag].append(w)

            for pos, words in pos2words.items():
                # filter rare words

                counter = Counter(words)
                words = set([w for w, count in counter.items() if count > min_occurrence])
                pos2words[pos] = words

            with open(pos2words_filename, 'wb') as f:
                pickle.dump(pos2words, f)

        return pos2words

    def _get_pos_tags(self, sentence: List[str]) -> List[str]:

        doc = spacy.tokens.Doc(vocab=self.nlp.vocab, words=sentence)
        for name, proc in self.nlp.pipeline:
            doc = proc(doc)
        pos_tags = [token.tag_ for token in doc]
        return pos_tags

    def get_equivalent_sentences(self, original_sentence: List[str], ) -> List[List[str]]:

        pos_tags = self._get_pos_tags(original_sentence)
        equivalent_sentences = [original_sentence]

        for i in range(self.num_sentences):

            sentence = []

            for j, (w, pos_tag) in enumerate(zip(original_sentence, pos_tags)):

                if pos_tag in self.pos_tags_to_replace and len(self.pos2words[pos_tag]) > 0:

                    sentence.append(random.choice(list(self.pos2words[pos_tag])))
                else:

                    sentence.append(w)

            equivalent_sentences.append(sentence)

        return equivalent_sentences


class EmbeddingBasedGenerator(EquivalentSentencesGenerator):

    def __init__(self, data_filename, num_sentences, topn=13):

        super().__init__(data_filename, num_sentences)

        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(utils.DEFAULT_PARAMS["word2vec"], binary=True)
        self.word_list = list(self.embeddings.wv.vocab)
        # self.knn = spatial.KDTree(self.embeddings.vectors[:1000])
        ngtpy.create(b"w2v", 300)
        self.knn = ngtpy.Index(b"w2v")
        self.knn.batch_insert(self.embeddings.vectors)
        self.knn.save()

        self.topn = topn

    def get_equivalent_sentences(self, original_sentence: List[str]) -> List[List[str]]:

        equivalent_sentences = []

        for i in range(self.num_sentences):

            sentence = []

            for j, w in enumerate(original_sentence):

                if (w in utils.DEFAULT_PARAMS["function_words"]) or (w not in self.word_list):

                    sentence.append(w)
                else:

                    # options = self.embeddings.most_similar(positive = [w], topn = self.topn)
                    # replacement, _ = random.choice(options)

                    # _, k_nearest = self.knn.query(self.embeddings[w], k=5)
                    # sentence.append(self.embeddings.vectors[replacement])

                    result = self.knn.search(self.embeddings[w], 3)
                    k_nearest = [x[0] for x in result]
                    replacement = random.choice(k_nearest)
                    sentence.append(self.word_list[replacement])

            equivalent_sentences.append(sentence)

        return equivalent_sentences