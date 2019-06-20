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
from functools import lru_cache

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class POSBasedEGenerator(EquivalentSentencesGenerator):

    def __init__(self, data_filename, output_file, pos_tags_to_replace, num_sentences,
                 pos2words_file):

        super().__init__(data_filename, output_file, num_sentences)

        self.nlp = spacy.load('en_core_web_sm')
        self.data_filename = data_filename
        self.pos2words_file = pos2words_file
        self.pos2words = self._get_POS2words_mapping()
        self.pos_tags_to_replace = pos_tags_to_replace

    def _get_POS2words_mapping(self, min_occurrence=50) -> DefaultDict[str, set]:
        """
        Iterate over the dataset, and find the words belonging to each POS tag.
        return: pos2words, a dictionary mapping pos tags (strings) to sets of words.
        """

        pos2words_filename = self.pos2words_file

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

                if (pos_tag in self.pos_tags_to_replace) and (len(self.pos2words[pos_tag]) > 0) and (w not in utils.DEFAULT_PARAMS['function_words']):
                    sentence.append(random.choice(list(self.pos2words[pos_tag])))
                else:

                    sentence.append(w)

            equivalent_sentences.append(sentence)

        return equivalent_sentences


class EmbeddingBasedGenerator(EquivalentSentencesGenerator):

    def __init__(self, data_filename, output_file, num_sentences, w2v_file, topn=13):

        super().__init__(data_filename, output_file, num_sentences)

        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        self.word_set = set(list(self.embeddings.wv.vocab))
        self.word_list = list(self.embeddings.wv.vocab)

        self.topn = topn

    @lru_cache(maxsize=None)
    def get_knn(self, w: str) -> List[str]:
        if (w in utils.DEFAULT_PARAMS['function_words']) or (w not in self.word_set):
            return [w]
        else:
            return [x[0] for x in self.embeddings.most_similar(positive=[w], topn=self.topn)]

    def get_equivalent_sentences(self, original_sentence: List[str]) -> List[List[str]]:

        equivalent_sentences = []

        for i in range(self.num_sentences):

            sentence = []

            for j, w in enumerate(original_sentence):

                k_nearest = self.get_knn(w)
                replacement = random.choice(k_nearest)
                sentence.append(replacement)

            equivalent_sentences.append(sentence)

        return equivalent_sentences

class BertGenerator(EquivalentSentencesGenerator):


        
    def __init__(self, data_filename, output_file, num_sentences):

        super().__init__(data_filename, output_file, num_sentences)
        

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.model.to("cuda")
        
    def _tokenize(self, original_sentence: List[str]) -> Tuple[List[str], Dict[int, int]]:
    
        bert_tokens = ["[CLS]"]
        orig_to_tok_map = {}
        
        for i, w in enumerate(original_sentence):
            
            orig_to_tok_map[i] = len(bert_tokens)
            tokenized_w = self.tokenizer.tokenize(w)
            bert_tokens.extend(tokenized_w)
           
        
        bert_tokens.append("[SEP]")
        
        return (bert_tokens, orig_to_tok_map)


    def get_equivalent_sentences(self, original_sentence: List[str]) -> List[List[str]]:

        bert_tokens, orig_to_tok_map = self._tokenize(original_sentence)

        equivalent_sentences = []

        for i in range(self.num_sentences):

            sentence = []

            for j, w in enumerate(original_sentence):

                if w in utils.DEFAULT_PARAMS["function_words"]:
                
                    sentence.append(w)
                    
                else:
                
                    masked_tokens = bert_tokens.copy()
                    masked_index = orig_to_tok_map[j]

                    
                    masked_tokens[masked_index] = "[MASK]"
                    print(masked_tokens)
                    indexed_tokens = self.tokenizer.convert_tokens_to_ids(masked_tokens)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    tokens_tensor = tokens_tensor.to('cuda')
                    
                    with torch.no_grad():
                         predictions = self.model(tokens_tensor)

                         predicted_indices = torch.argsort(predictions[0, masked_index])[-5:].cpu().numpy()
                         print(self.tokenizer.convert_ids_to_tokens(predicted_indices)

                    

            equivalent_sentences.append(sentence)

        return equivalent_sentences
