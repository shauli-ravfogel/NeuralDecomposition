import spacy
import utils
from typing import DefaultDict, List, Tuple, Dict
from collections import defaultdict, Counter
import random
import os.path
import tqdm
import pickle
import gensim
from functools import lru_cache
import numpy as np
import nltk

import torch
from pytorch_transformers import BertTokenizer, BertForMaskedLM


class EquivalentSentencesGenerator:
    """
    An abstract class for generating syntactically-equivalent sentenes, that is, sentences sharing
    the same underlying hierarchical structure (but possibly different semantics).
    """

    def __init__(self, data_filename: str, output_file: str,
                 num_sentences: int):
        self.sentences = utils.read_sentences(data_filename)
        self.num_sentences = num_sentences
        self.output_file = output_file

    def generate(self) -> Dict[int, List[List[str]]]:
        sents_dict = dict()
        print("Generating equivalent sentences...")

        for i, sentence in tqdm.tqdm(enumerate(self.sentences), total=len(self.sentences)):
            equivalent_sentences = self.get_equivalent_sentences(sentence)
            sents_dict[i] = equivalent_sentences

        if (i % 100 == 0 and i != 0) or (i == len(self.sentences) - 1):
            with open(self.output_file, 'wb') as f:
                pickle.dump(sents_dict, f)

        return sents_dict

    def get_equivalent_sentences(self, original_sentence: List[str]) -> List[List[str]]:
        raise NotImplementedError()


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

                if (pos_tag in self.pos_tags_to_replace) and (len(self.pos2words[pos_tag]) > 0) and (
                        w not in utils.DEFAULT_PARAMS['function_words']):

                    sentence.append(random.choice(list(self.pos2words[pos_tag])))
                else:

                    sentence.append(w)

            equivalent_sentences.append(sentence)

        return equivalent_sentences


class POSBasedEGenerator2(EquivalentSentencesGenerator):

    def __init__(self, data_filename, output_file, pos_tags_to_replace, num_sentences,
                 pos2words_file, order=1):

        super().__init__(data_filename, output_file, num_sentences)

        self.order = order
        self.nlp = spacy.load('en_core_web_sm')
        self.data_filename = data_filename
        self.pos2words_file = pos2words_file
        self.pos2words = self._get_POS2words_mapping()
        self.pos_tags_to_replace = pos_tags_to_replace

    def _pad(self, sentence, pos_tags):

        for i in range(self.order):
            sentence.append("<E>")
            pos_tags.append("<E>")
            sentence.insert(0, "<S>")
            pos_tags.insert(0, "<S>")

        return sentence, pos_tags

    def _get_ngrams(self, sentence, pos_tags):

        words_ngrams = [sentence[i - self.order: i + self.order + 1] for i in
                        range(self.order, len(sentence) - self.order)]
        pos_ngrams = [pos_tags[i - self.order: i + self.order + 1] for i in
                      range(self.order, len(sentence) - self.order)]

        return words_ngrams, pos_ngrams

    def _get_POS2words_mapping(self, min_occurrence=20) -> DefaultDict[str, set]:
        """
        Iterate over the dataset, and find the words belonging to each POS tag.
        return: pos2words, a dictionary mapping pos tags (strings) to sets of words.
        """

        pos2words_filename = self.pos2words_file

        if os.path.isfile(pos2words_filename) and True:

            with open(pos2words_filename, 'rb') as f:
                pos2words = pickle.load(f)

        else:

            print("Collecting POS:words mapping...")

            pos2words = defaultdict(set)
            sentences = utils.read_sentences(self.data_filename)
            all_words = [w for sent in sentences for w in sent]
            words_counter = Counter(all_words)
            common_words = set([w for w, count in words_counter.items() if count > min_occurrence])

            for sentence in tqdm.tqdm(sentences):

                pos_tags = self._get_pos_tags(sentence)

                # add dummy characters at the end

                sentence, pos_tags = self._pad(sentence, pos_tags)

                # collect occurrences of POS ngrams

                words_ngrams, pos_ngrams = self._get_ngrams(sentence, pos_tags)

                for i, (w_ngrams, pos_tag_ngrams) in enumerate(zip(words_ngrams, pos_ngrams)):

                    w, pos = w_ngrams[self.order], pos_tag_ngrams[self.order]

                    if w in common_words:
                        pos2words["*".join(pos_tag_ngrams)].add(w)

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

        sentence, pos_tags = self._pad(original_sentence[:], pos_tags)

        equivalent_sentences = [original_sentence]
        words_ngrams, pos_ngrams = self._get_ngrams(sentence, pos_tags)

        for i in range(self.num_sentences):

            sentence = []

            for j, (w_ngrams, pos_tag_ngrams) in enumerate(zip(words_ngrams, pos_ngrams)):

                pos_tag, w = pos_tag_ngrams[self.order], w_ngrams[self.order]
                pos_seq = "*".join(pos_tag_ngrams)

                if (pos_tag in self.pos_tags_to_replace) and (pos_seq in self.pos2words) and (
                        len(self.pos2words[pos_seq]) > 0) and (w not in utils.DEFAULT_PARAMS['function_words']):

                    sentence.append(random.choice(list(self.pos2words[pos_seq])))
                else:

                    sentence.append(w)

            equivalent_sentences.append(sentence)

        return equivalent_sentences


class EmbeddingBasedGenerator(EquivalentSentencesGenerator):

    def __init__(self, data_filename, output_file, num_sentences, w2v_file, topn=8):

        super().__init__(data_filename, output_file, num_sentences)

        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        self.word_set = set(list(self.embeddings.wv.vocab))
        self.word_list = list(self.embeddings.wv.vocab)

        self.topn = topn

    @lru_cache(maxsize=256)
    def get_knn(self, w: str) -> List[str]:
        if (w in utils.DEFAULT_PARAMS['function_words']) or (w not in self.word_set):
            return [w]
        else:
            return [x[0] for x in self.embeddings.most_similar(positive=[w], topn=self.topn)]

    def get_equivalent_sentences(self, original_sentence: List[str]) -> List[List[str]]:

        equivalent_sentences = [original_sentence]

        for i in range(self.num_sentences):

            sentence = []

            for j, w in enumerate(original_sentence):
                k_nearest = self.get_knn(w)
                replacement = random.choice(k_nearest)
                sentence.append(replacement)

            equivalent_sentences.append(sentence)

        return equivalent_sentences


class BertGenerator(EquivalentSentencesGenerator):

    def __init__(self, data_filename, output_file, num_sentences, topn=8, ignore_first_k=0, maintain_pos=False,
                 cuda_device=0):

        super().__init__(data_filename, output_file, num_sentences)

        self.cuda_device = cuda_device
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        self.model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
        
        self.model.eval()
        self.model.to('cuda:{}'.format(self.cuda_device))
        self.forbidden_guesses = utils.DEFAULT_PARAMS["function_words"]
        self.topn = topn
        self.ignore_first_k = ignore_first_k
        self.maintain_pos = maintain_pos

        if self.maintain_pos:

            # Load nltk tagger

            from nltk.corpus import brown
            brown_tagged_sents = brown.tagged_sents()
            brown_sents = brown.sents()
            self.unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)

            # Load spaCy tagger

            self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner", "sentencizer"])

            print("Collecting word:POS mapping...")
            self.w2pos = defaultdict(Counter)
            for i, sentence in tqdm.tqdm(enumerate(self.sentences), total=len(self.sentences)):

                tags = self._get_pos_tags(sentence)
                for w, t in zip(sentence, tags):
                    self.w2pos[w][t] += 1

    def choose_word(self, guesses, original_pos=None):

        if original_pos == "VBP": original_pos = "VB"  # use use a unigram tagger, so VBPs are tagged as VBs

        guesses = [w for w in guesses if "##" not in w]

        if self.maintain_pos and original_pos is not None:

            # guesses = [w for w, pos in self.unigram_tagger.tag(guesses) if (w not in self.forbidden_guesses) and (original_pos == pos)]
            guesses = [w for w in guesses if (self.w2pos[w][original_pos] > 1) and (w not in self.forbidden_guesses)]

        else:
            guesses = [w for w in guesses if w not in self.forbidden_guesses]

        if self.ignore_first_k:
            guesses = guesses[self.ignore_first_k:]

        guesses = guesses[:self.topn]

        if not guesses:
            return None

        return random.choice(guesses)

    def _tokenize(self, original_sentence: List[str]) -> Tuple[List[str], Dict[int, int]]:

        """
        Parameters
        ----------
        Returns
        -------
        bert_tokens: The sentence, tokenized by BERT tokenizer.
        orig_to_tok_map: An output dictionary consisting of a mapping (alignment) between indices in the original tokenized sentence, and indices in the sentence tokenized by the BERT tokenizer. See https://github.com/google-research/bert
        """

        bert_tokens = ["[CLS]"]
        orig_to_tok_map = {}
        has_subwords = False
        is_subword = []

        for i, w in enumerate(original_sentence):
            tokenized_w = self.tokenizer.tokenize(w)
            has_subwords = len(tokenized_w) > 1
            is_subword.append(has_subwords)
            bert_tokens.extend(tokenized_w)

            orig_to_tok_map[i] = len(bert_tokens) - 1

        bert_tokens.append("[SEP]")

        return (bert_tokens, orig_to_tok_map)

    def _get_pos_tags(self, sentence: List[str]) -> List[str]:

        doc = spacy.tokens.Doc(vocab=self.nlp.vocab, words=sentence)
        for name, proc in self.nlp.pipeline:
            doc = proc(doc)
        pos_tags = [token.tag_ for token in doc]
        return pos_tags

    def get_equivalent_sentences(self, original_sentence: List[str], online=False, topn=6) -> List[List[str]]:

        raise NotImplementedError


class IndependentBertGenerator(BertGenerator):

    def __init__(self, data_filename, output_file, num_sentences, topn=8):
        super().__init__(data_filename, output_file, num_sentences, topn=topn)

    def get_equivalent_sentences(self, original_sentence: List[str], online=False, topn=6) -> List[List[str]]:

        equivalent_sentences = [original_sentence]

        bert_tokens, orig_to_tok_map = self._tokenize(original_sentence)
        # print(orig_to_tok_map)
        options = []  # a list of list, containing Bert's guesses for each position in the sentence.

        for j, w in enumerate(original_sentence):

            if (w in utils.DEFAULT_PARAMS["function_words"]):

                options.append([w])

            else:

                masked_tokens = bert_tokens.copy()
                masked_index = orig_to_tok_map[j]  # find the index of the curent word in the BERT-tokenized sentence

                # check if BERT tokenizer has splitted the original word to subwrods.

                subwords_exist = (j != len(original_sentence) - 1) and (orig_to_tok_map[j + 1] - orig_to_tok_map[j]) > 1

                masked_tokens[masked_index] = "[MASK]"

                # convert the masked sentence to a tensor.

                indexed_tokens = self.tokenizer.convert_tokens_to_ids(masked_tokens)
                tokens_tensor = torch.tensor([indexed_tokens])
                tokens_tensor = tokens_tensor.to('cuda:{}'.format(self.cuda_device))

                with torch.no_grad():

                    predictions = self.model(tokens_tensor)

                    predicted_indices = torch.argsort(predictions[0, masked_index])[:].cpu().numpy()
                    guesses = self.tokenizer.convert_ids_to_tokens(predicted_indices)
                    guesses = list(filter(lambda w: w not in self.forbidden_guesses, guesses))

                    if guesses == []:
                        guesses = [w]

                    # if the word was splitted into subwords, we heuristically mask the first subword only
                    # the guessed word is then suffixed with the (unmodified) remaining subwords.

                    if subwords_exist:
                        suffix = bert_tokens[masked_index + 1: orig_to_tok_map[j + 1]]
                        suffix_str = "".join(suffix)
                        guesses = [w + suffix_str for w in guesses]

                options.append(guesses)

        # randomly generate the equivalent sentences from BERT's guesses

        for i in range(self.num_sentences):

            sentence = []

            for j in range(len(original_sentence)):
                sentence.append(random.choice(options[j]).replace("##", ""))

            equivalent_sentences.append(sentence)

        return equivalent_sentences


class OnlineBertGenerator(BertGenerator):

    def __init__(self, data_filename, output_file, num_sentences, topn=9, ignore_first_k=2):

        super().__init__(data_filename, output_file, num_sentences, topn=topn, ignore_first_k=ignore_first_k)

    def get_equivalent_sentences(self, original_sentence: List[str], online=False, topn=6) -> List[List[str]]:

        equivalent_sentences = [original_sentence]

        for i in range(self.num_sentences):

            bert_tokens, orig_to_tok_map = self._tokenize(original_sentence)
            sentence = []

            tokens_tensor = torch.zeros(1, len(bert_tokens), dtype=torch.long)
            tokens_tensor = tokens_tensor.to('cuda:{}'.format(self.cuda_device))

            for j, w in enumerate(original_sentence):

                if w in utils.DEFAULT_PARAMS["function_words"]:

                    sentence.append(w)

                else:

                    masked_tokens = bert_tokens  # .copy()
                    masked_index = orig_to_tok_map[j]
                    subwords_exist = (j != len(original_sentence) - 1) and (
                                orig_to_tok_map[j + 1] - orig_to_tok_map[j]) > 1

                    masked_tokens[masked_index] = "[MASK]"

                    indexed_tokens = self.tokenizer.convert_tokens_to_ids(masked_tokens)
                    tokens_tensor = tokens_tensor.copy_(torch.tensor([indexed_tokens]))

                    with torch.no_grad():

                        predictions = self.model(tokens_tensor)

                        _, predicted_indices = torch.topk(predictions[0, masked_index], k=40, sorted=True, largest=True)
                        guesses = self.tokenizer.convert_ids_to_tokens(predicted_indices.cpu().numpy())[::-1]

                        w = self.choose_word(guesses)
                        bert_tokens[masked_index] = w  # update the sentence with the word chosen.

                        if subwords_exist:
                            suffix = bert_tokens[masked_index + 1: orig_to_tok_map[j + 1]]
                            suffix_str = "".join(suffix)
                            w += suffix_str

                        sentence.append(w.replace("##", ""))

            equivalent_sentences.append(sentence)

        return equivalent_sentences


class BatchedOnlineBertGenerator(BertGenerator):

    def __init__(self, data_filename, output_file, num_sentences, topn=10, ignore_first_k=0, maintain_pos=False,
                 cuda_device=0):

        super().__init__(data_filename, output_file, num_sentences, topn=topn, ignore_first_k=ignore_first_k,
                         maintain_pos=maintain_pos, cuda_device=cuda_device)

    def get_equivalent_sentences(self, original_sentence: List[str], online=False, topn=6) -> List[List[str]]:

        bert_tokens, orig_to_tok_map = self._tokenize(original_sentence)

        batch_bert_tokens = np.empty((self.num_sentences, len(bert_tokens)), dtype=object)

        batch_bert_tokens[:, ] = bert_tokens.copy()

        equivalent_sentences = np.empty((self.num_sentences, len(original_sentence)), dtype=object)
        equivalent_sentences[0, :] = original_sentence.copy()
        tokens_tensor = torch.zeros((self.num_sentences, len(bert_tokens)), dtype=torch.long)
        tokens_tensor = tokens_tensor.to('cuda:{}'.format(self.cuda_device))

        if self.maintain_pos:
            original_pos_tags = self._get_pos_tags(original_sentence)

        indices_and_words = list(enumerate(original_sentence.copy()))

        for j, w in indices_and_words:

            if w in utils.DEFAULT_PARAMS["function_words"]:
                equivalent_sentences[:, j].fill(w)

            else:

                masked_index = orig_to_tok_map[j]
                original_pos = original_pos_tags[j] if (self.maintain_pos) else None
                batch_bert_tokens[:, masked_index] = "[MASK]"
                indexed_tokens = np.empty_like(batch_bert_tokens, dtype=int)

                for i in range(self.num_sentences):
                    indexed_tokens[i, :] = self.tokenizer.convert_tokens_to_ids(batch_bert_tokens[i])

                tokens_tensor = tokens_tensor.copy_(torch.from_numpy(indexed_tokens))

                with torch.no_grad():

                    predictions = self.model(tokens_tensor)[0]  # (num_sentences, len(bert_tokens), voc_size)

                _, predicted_indices = torch.topk(predictions[:, masked_index, :], k=220, sorted=True, largest=True,
                                                  dim=-1)
                predicted_indices = predicted_indices.cpu().numpy()  # (num_sentences, k)

                for i in range(1, self.num_sentences):  # the first sentence remains the original one

                    guesses = self.tokenizer.convert_ids_to_tokens(predicted_indices[i])  # (k,)
                    chosen_w = self.choose_word(guesses, original_pos=original_pos)

                    if chosen_w is not None:
                        batch_bert_tokens[i, masked_index] = chosen_w

                        """
                                        if subwords_exist:
                     
                                                suffix = batch_bert_tokens[i, masked_index + 1: orig_to_tok_map[j + 1]]
                                                suffix_str = "".join(suffix)
                                                chosen_w += suffix_str
                                        """

                    equivalent_sentences[i, j] = chosen_w.replace("##",
                                                                  "") if chosen_w is not None else w  # update the sentence with the word chosen.
        return equivalent_sentences

# PARAMS WERE 100, 9, IGNORE 2
