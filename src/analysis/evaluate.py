import sys

sys.path.append('src/generate_dataset')
from utils import DEFAULT_PARAMS

FUNCTION_WORDS = DEFAULT_PARAMS["function_words"]
sys.path.append('src/analysis/tree_distance')
import tree_similarity
from typing import List, Tuple, Dict
import typing
from syntactic_extractor import SyntacticExtractor
# import tree_similarity
import sklearn
from sklearn import metrics
import numpy as np
from tqdm.auto import tqdm
import spacy
import random
from scipy.stats.stats import pearsonr
from scipy.stats import entropy
from collections import Counter, defaultdict
import copy
from annoy import AnnoyIndex
import os
import nltk


Sentence_vector = typing.NamedTuple("Sentence_vector",
                                    [('sent_vectors', np.ndarray), ('sent_str', List[str]),
                                     ("doc", spacy.tokens.Doc), ("tree", nltk.Tree)])
Word_vector = typing.NamedTuple("Word_vector", [('word_vector', np.ndarray), ('sentence', List[str]),
                                                ("doc", spacy.tokens.Doc), ("index", int), ("tree", nltk.Tree)])


def run_tests(sentence_reprs: List[Sentence_vector], extractor, num_queries, method, num_words,
              ignore_function_words=True):
    """
        the main function for running the experiments & printing their results.
        ----------
        Parameters
        embds_and_sents: A list of tuples (sentence_vectors ,sentence_str).
        extractor: SyntacticExtractor, required.
                   An instance of the interface SyntacticExtractor that extracts syntactic representations.
        ----------
        Returns
        -------
        Reports results in the terminal. Produces the following txt files:
              closest_sentences.extractor:False.txt: contains the results of the closest-sentence test, berfore the application of the syntactic extractor. Format: query-sentence tab closest-sentence tab kernel-similarity tab edit-distance-similarity:
              closest_sentences.extractor:True.txt: as above, after the applciation of the syntactic extractor.
              closest_words.extractor:False.txt: contains the results of the closest-word test, berfore the application of the syntactic extractor. Format: query-word tab closest-word tab query-word-dep tab closest-word-dep tab correct.
              The full context of the sentence is provided, and the words are marked by ***asterisks***.
              losest_words.extractor:True.txt: as above, after the application of the syntactic extractor.
        """

    words_reprs = sentences2words(sentence_reprs, num_words=num_words,
                                  ignore_function_words=ignore_function_words)  # words representatons

    # closest-word, with ELMO alone (basline)
    closest_word_test(words_reprs, num_queries=num_queries, method=method, extractor=None)

    # closest-word, with ELMO + syntactic extractor
    closest_word_test(words_reprs, num_queries=num_queries, method=method, extractor=extractor)
    print('all words used: {}'.format(len(words_reprs)))
    hard_pos = get_hard_pos(sentence_reprs)
    hard_words = split_pos(words_reprs, hard_pos[:5])

    print('hard words')
    print('hard pos word useds: {}'.format(len(hard_words)))
    print('baseline')
    closest_word_test(hard_words, num_queries=num_queries, method=method, extractor=None)
    print('our method')
    closest_word_test(hard_words, num_queries=num_queries, method=method, extractor=extractor)

    short_sen, long_sen = length_split(sentence_reprs)
    for split, name in zip([sentence_reprs, short_sen, long_sen], ['all', 'short', 'long']):
        print('running sentence eval on: {} split, num sen: {}'.format(name, len(split)))
        # closest-sentence, with ELMO alone (basline)
        closest_sentence_test(split, num_queries=num_queries, method=method, extractor=None)

        # closest-sentence, with ELMO + syntactic extractor
        closest_sentence_test(split, num_queries=num_queries, method=method, extractor=extractor)






def choose_words_from_sents(sent_reprs, extractor, n = 10000):

    sents_data = random.choices(sent_reprs, k=n)

    # Apply syntactic extractor

    if extractor is not None:

        print("Applying syntactic extractor...")

        for i, sent_repr in tqdm(enumerate(sents_data), total=len(sents_data), ascii=True):
            sents_data[i] = sent_repr._replace(
                sent_vectors = extractor.extract(sent_repr.sent_vectors))

    # choose words
    data = []

    for sent in sents_data:

        i = np.random.choice(range(len(sent.sent_str)))
        vec = sent.sent_vectors[i].reshape(-1)
        word = sent.sent_str[i]
        words = sent.sent_str
        doc = sent.doc
        data.append(Word_vector(vec.copy(), words, doc, i))

    return data





def syntax_neutralization(sentence_representations: List[Sentence_vector], num_queries, extractor, alpha = 5):

    # collect scores on unmodified vectors. Those are assumed to capture mainly semantics.

    values = [np.mean(sent_repr.sent_vectors, axis = 0) for sent_repr in sentence_representations]
    queries = values[:num_queries]
    dists_original = sklearn.metrics.pairwise_distances(queries, values, metric="euclidean")

    # collect scores on modified vectors.

    values = copy.deepcopy(sentence_representations)
    for i, sent in enumerate(values):
        values[i] = np.mean(extractor.extract(sent.sent_vectors), axis = 0)

    queries = values[:num_queries]
    dists_after = sklearn.metrics.pairwise_distances(queries, values, metric="euclidean")

    dists_total = dists_original - alpha * dists_after
    sents = np.array([sentence_representations[i].sent_str for i in range(len(sentence_representations))], dtype = "object")
    k = 5
    top_k = dists_total.argsort(axis=1)[:, :k + 1]
    closest_indices = top_k[:, 0: k]

    with open("sents.txt", "w", encoding = "utf-8") as f:
      for i in range(num_queries):

        original = " ".join(sents[i])
        f.write(original + "\n")
        f.write("========================================\n")
        closest_idx = closest_indices[i]
        closest_sents = sents[closest_idx]

        for j in range(k):

            f.write(" ".join(closest_sents[j]) + "\n")
            f.write("-------------------------------------\n")




def split_pos(words_reprs: List[Word_vector], hard_pos: List[str]) -> List[Word_vector]:
    filtered_words = []

    for word in words_reprs:
        pos = word.doc[word.index].pos_
        if pos in hard_pos:
            filtered_words.append(word)
    return filtered_words


def get_hard_pos(sentence_representations: List[Sentence_vector]) -> List[str]:
    pos_dep = defaultdict(list)
    for sentence in sentence_representations:
        doc = sentence.doc
        for w in doc:
            pos_dep[w.pos_].append(w.dep_)

    norm_dic = {}
    # normalizing
    for k, v in pos_dep.items():
        temp_dic = {}
        d = Counter(v)
        factor = 1.0 / sum(d.values())
        for kk, val in d.items():
            temp_dic[kk] = val * factor
        norm_dic[k] = temp_dic

    # calculating entropy
    dic = {}
    for k, v in norm_dic.items():
        probs = list(v.values())
        # print(k, probs)
        ent = entropy(probs)
        dic[k] = ent

    ordered_ent = sorted(dic.items(), key=lambda kv: kv[1])
    ordered_ent.reverse()
    print(ordered_ent)
    return [x[0] for x in ordered_ent]


def length_split(sentence_representations: List[Sentence_vector]) -> List[List[Sentence_vector]]:
    short_sen, long_sen = [], []
    for sentence in sentence_representations:
        if len(sentence.sent_str) < 15:
            short_sen.append(sentence)
        else:
            long_sen.append(sentence)
    return [short_sen, long_sen]


def persist_for_tsne(word_reprs, extractor, n=10000):
    def to_string(np_array):
        return "\t".join(["%0.4f" % x for x in np_array])

    def sentence2str(sentence: List[str], index: int):

        return " ".join(sentence[:index] + ["@@@" + sentence[index] + "@@@"] + sentence[index + 1:])

    data = random.choices(word_reprs, k=n)

    # Apply syntactic extractor

    if extractor is not None:

        print("Applying syntactic extractor...")

        for i, word_representation in tqdm(enumerate(data), total=len(data), ascii=True):
            data[i] = word_representation._replace(
                word_vector=extractor.extract(word_representation.word_vector).reshape(-1)[:])

    labels, vecs = [], []

    for i in range(n):
        x = data[i]
        vec = x.word_vector
        position = x.index
        token = x.sentence[position]
        sent = x.sentence
        dep_edge = x.doc[x.index].dep_
        parent_dep = x.doc[x.index].head.dep_
        pos = x.doc[x.index].tag_
        depth = node_height(x.doc[x.index])

        vec = to_string(vec)
        word_labels = "\t".join(
            [str(position), pos, dep_edge, parent_dep, str(depth), token, sentence2str(sent, position)])
        labels.append(word_labels)
        vecs.append(vec)

    with open("vecs.tsv", "w") as f:

        for v in vecs:
            f.write(v + "\n")

    with open("labels.tsv", "w") as f:

        f.write("position\tPOS\tdep-edge\tparent's dep-edge\tdep-tree-depth\ttoken\tsent\n")
        for word_labels in labels:
            f.write(word_labels + "\n")

            
def gat_constituency_path_to_root(tree: nltk.Tree, leaf_index: int) -> List[str]:
    
    parented_tree = nltk.tree.ParentedTree.convert(tree)
    labels = []
    path_to_leaf = parented_tree.leaf_treeposition(leaf_index)
    path_to_leaf_POS = path_to_leaf[:-1]
    
    current, is_root = parented_tree[path_to_leaf_POS], False
    
    while current is not None:
        
        labels.append(current.label())
        current = current.parent()
        
    return labels[:-1]
  
  

def get_path_to_root(word: Word_vector):
    word = word.doc[word.index]
    token_deps = [word.dep_]
    curr_depth = 0

    while word.head != word:  # while not root

        head_dep = word.head.dep_
        token_deps.append(head_dep)
        word = word.head
        curr_depth += 1
    return token_deps


color_by_dep = lambda word: word.doc[word.index].dep_
color_by_depth = lambda word: len(get_path_to_root(word))


def color_by_dep_in_path(word, label="advcl"):
    path_to_root = get_path_to_root(word)
    print(label, path_to_root, label in path_to_root)

    if label in path_to_root:
        return "dep {} in path".format(label)
    else:
        return "dep {} not in path".format(label)


def test_demo_words(all_word_reprs, elmo_embedder, extractor):
    nlp = spacy.load('en_core_web_sm')
    sentence = "had i not seen it myself, i could not have believed that."
    sentence = nlp(sentence)
    index = 3
    word = sentence[index].text
    print("word: {}".format(word))
    closest = get_closest_word_demo(all_word_reprs, sentence, index, elmo_embedder, extractor, k=10, method="cosine")

    for value_word_repr in closest:
        ind = value_word_repr.index
        sent = value_word_repr.sentence
        w = value_word_repr.word
        value_str = " ".join(sent[:ind] + ["***" + w + "***"] + sent[ind + 1:])
        print(value_str)
        print("---------------------------------------------")


def test_demo_sentences(all_sents_repr, elmo_embedder, extractor):
    nlp = spacy.load('en_core_web_sm')
    sentence = "had i not seen it myself, i could not have believed that."
    sentence = nlp(sentence)
    closest = get_closest_sentence_demo(all_sents_repr, sentence, elmo_embedder, extractor, k=10, method="cosine")

    for value_sent_repr in closest:
        sent = value_sent_repr.sent_str
        print(" ".join(sent))
        print("---------------------------------------------")


def get_closest_word_demo(all_word_reprs: List[Word_vector], sentence: spacy.tokens.Doc,
                          index: int, embedder, extractor, k: int = 5, method: str = 'l2') -> List[Word_vector]:
    """
    Parameters
    ----------
    all_word_reprs: A list of Word_vector objects, on which the closest-vector query is performed.
    sentence: a spacy Doc representing the input sentence.
    index: the index in the sentence to query.
    embedder: embedder object
    extractor: syntactic_extractor object. If None, don't apply syntactic extractor
    k: int, how many closest-neighbors to collect.
    method: "cosine" / "l2"
    ----------
    Returns
    -------
    closest: A list of Word_vector objects, representing the k closest-vectors to the query vector.
   """

    sent_words = [token.text for token in sentence]
    sent_vecs, _ = embedder.run_embedder([sent_words])[0]

    query_vec = sent_vecs[index]

    all_vecs = [word_repr.word_vector for word_repr in all_word_reprs]
    all_sents = [word_repr.sentence for word_repr in all_word_reprs]

    if extractor is not None:
        print("applying syntactic extractor")
        query_vec = extractor.extract(query_vec)

    closest = get_closest_vectors(np.array(all_vecs), query_vec.reshape(1, -1),
                                  all_sents, method=method, k=k,
                            ignore_same_vec=False)[0]
    return [all_word_reprs[ind] for ind in closest]


def get_closest_sentence_demo(all_sentence_np: List[np.ndarray], all_sentence: List[Sentence_vector],
                              sentence_vec: np.ndarray, extractor, k: int = 5, method: str = 'l2') -> List[
    Sentence_vector]:
    """
    Parameters
    ----------
    all_word_reprs: A list of Word_vector objects, on which the closest-vector query is performed.
    sentence_vec: a list of np array values.
    index: the index in the sentence to query.
    extractor: syntactic_extractor object. If None, don't apply syntactic extractor
    k: int, how many closest-neighbors to collect.
    method: "cosine" / "l2"
    ----------
    Returns
    -------
    closest: A list of Sentence_vector objects, representing the k closest-vectors to the query vector.
   """

    if extractor is not None:
        sentence_vec = extractor.extract(sentence_vec)

    # query_mean = np.mean(sentence_vec, axis=0, keepdims=True)
    sents = [s.sent_str for s in all_sentence]

    closest = get_closest_vectors(all_sentence_np, sentence_vec, sents, method=method, k=k, ignore_same_vec=False)[0]
    return [all_sentence[ind] for ind in closest]


def parse(sentences: List[List[str]], batch_size = 5000) -> Tuple[List[spacy.tokens.Doc], List[nltk.Tree]]:
    """
        Parameters
        sentences: A list of sentence, where each sentence is a list of word strings.
        ----------
        Returns
        -------
        all_deps: A list of dependency edges, where all_deps[i][j] is the dep of word j in sentence i.
        """

    print("Parsing...")
 
    nlp = spacy.load('en_core_web_sm')
    nlp.remove_pipe("ner")
    
    print("Creating docs...")
    docs = [nlp.tokenizer.tokens_from_list(sent) for sent in tqdm(sentences, ascii = True)]
    
    pipeline = [(name, proc) for name, proc in nlp.pipeline]
    
    for name, component in pipeline:
        print("Applying {}...".format(name))
        docs = component.pipe(docs, batch_size = batch_size)
    
    docs = list(docs)
    import tensorflow as tf
    import benepar
    parser = benepar.Parser("benepar_en2")
    print("Running benepar parser...")
    with tf.device('/gpu:0'):

        trees = list(parser.parse_sents(sentences))
    
    return docs, trees


def get_closest_vectors_efficient(all_vecs: List[np.ndarray], queries: List[np.ndarray], sents: List[str],
                                  method: str, k=5, ignore_same_vec=True, filter_same_sentence=True):

    print('building index')
    indexer = AnnoyIndex(all_vecs[0].shape[0], 'angular')

    indexer_name = 'annoy_knn_' + str(all_vecs[0].shape[0]) + '_n:' + str(len(all_vecs))
    if os.path.exists(indexer_name):
        indexer.load(indexer_name)
    else:
        for i in tqdm(range(len(all_vecs))):
            indexer.add_item(i, all_vecs[i])

        indexer.build(100)  # 10 trees
        indexer.save(indexer_name)

    closest_indices = []
    for ind, query in enumerate(tqdm(queries)):
        # 100 is a random "large" number in order to be able to filter same sentences
        closest = indexer.get_nns_by_vector(query, k + 100 + 1)
        if filter_same_sentence:
            filtered_closest = []
            query_sent = sents[ind]
            for closest_ind in closest:
                if sents[closest_ind] == query_sent:
                    continue
                filtered_closest.append(closest_ind)
            closest = filtered_closest
        if ignore_same_vec:
            if np.array_equal(all_vecs[closest[0]], queries[ind]):
                closest.pop(0)
        closest_indices.append(closest[:k])

    return closest_indices


def get_closest_vectors(all_vecs: List[np.ndarray], queries: List[np.ndarray], sents: List[str], method: str, k=5,
                        ignore_same_vec=True, filter_same_sentence=True):
    if method == "cosine":

        # normalize the vectors
        all_vecs = all_vecs / np.linalg.norm(all_vecs, axis=1)[:, None]
        queries = queries / np.linalg.norm(queries, axis=1)[:, None]

        # perform dot product
        distances = sklearn.metrics.pairwise_distances(queries, all_vecs, metric="cosine")

    else:
        distances = sklearn.metrics.pairwise_distances(queries, all_vecs, metric="euclidean")

    if ignore_same_vec and filter_same_sentence:

        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if sents[i] == sents[j]:
                    distances[i, j] = 1e7

    top_k = distances.argsort(axis=1)[:, :k + 1]

    if ignore_same_vec:
        closest_indices = top_k[:, 1: k + 1]  # ignore the same vec
    else:
        closest_indices = top_k[:, 0: k]  # don't ignore the same vec

    return closest_indices


def get_sentence_representations(embds_and_sents: List[Tuple[List[np.ndarray], str]]) -> List[Sentence_vector]:
    """
        Parameters
        embds_and_sents: A list of tuples (sents_vectors, sentence string)
        ----------
        Returns
        embds_sents_deps: A list of Sentence_vectors.
      """

    embds, sentences = list(zip(*embds_and_sents))
    docs = parse(sentences)

    assert len(sentences) == len(embds) == len(docs)

    embds_sents_deps = [Sentence_vector(e, s, tok) for e, s, tok in
                        zip(embds, sentences, docs)]

    return embds_sents_deps


def sentences2words(sentence_representations: List[Sentence_vector],
                    num_words, ignore_function_words=True) -> List[Word_vector]:
    """
        Parameters
        sentence_representations: A list of Sentence_vector, required.
                        contains the representation of all sentences.
        num_words: int, required.
                How many words to collect.
        ignore_function_words: bool, optional.
                   whether or not to filter function words.
        ----------
        Returns
        -------
        word_vectors: A list of Word_vector, containing selected words from all sentences.
        """

    data = []

    print("Collecting words...")

    for i, sent_rep in tqdm(enumerate(sentence_representations), total=num_words):

        if len(data) > num_words: break

        vectors, words, doc = sent_rep

        for j, (vec, w) in enumerate(zip(vectors, words)):

            if ignore_function_words and w in FUNCTION_WORDS: continue

            data.append(Word_vector(vec.copy(), words, doc, j))

    random.seed(0)
    random.shuffle(data)

    print("Collected {} words".format(len(data)))
    return data


def closest_sentence_test(sentence_representations: List[Sentence_vector],
                          extractor=None, num_queries=15,
                          method="cosine"):
    """
        Parameters
        embds_sents_deps: A list of tuples (sents_vectors, sentence string, deps), required.
                        contains embeddings, sents and deps of all sentences.
        extractor: SyntacticExtractor, optional.
                   An instance of the interface SyntacticExtractor that extracts syntactic representations.
                   if None, use unmodified ELMO vectors. else, project each ELMO vectors using the extractor.
        num_queries:
                   how many closest-sentence queries to perform.
        method: str, optional (cosine / euc)
                what kind of similarity function to use.
        ----------
        """

    print("Performing closest-sentence test. Using extractor: {}".format(extractor is not None))

    # if supplied with extractor, use it to project the vectors to the syntactic space.

    # (TODO: this and the mean vector calculation should probably moved to a separate function [sentences2words?],
    #  as they are dealing with data processing, not test logic.)

    if extractor is not None:

        for i, sent in enumerate(sentence_representations):
            sentence_representations[i] = sent._replace(sent_vectors=extractor.extract(sent.sent_vectors))

    # represent each sentence as its mean vector

    for i, sent in enumerate(sentence_representations):
        sentence_representations[i] = sent._replace(sent_vectors=np.mean(sent.sent_vectors, axis=0))

    vecs, sents = [sent.sent_vectors for sent in sentence_representations], [sent.sent_str for sent in
                                                                             sentence_representations]
    vecs = np.array(vecs)

    # perform closest_vector query.

    queries = vecs[:num_queries]
    closest_indices = get_closest_vectors(vecs, queries, sents, method=method, k=1)

    query_sents = [sentence_representations[i] for i in range(num_queries)]
    value_sents = [sentence_representations[closest_ind[0]] for closest_ind in closest_indices]

    kernel_sims = tree_similarity.get_similarity_scores(query_sents, value_sents)
    avg_kernel_sim = np.mean(kernel_sims)
    #avg_edit_sims = np.mean(edit_sims)

    fname = "results/closest_sentences.extractor:{}.txt".format(extractor is not None)
    with open(fname, "w", encoding="utf8") as f:

        for (query, value, kernel_sim) in zip(query_sents, value_sents, kernel_sims):
            f.write(" ".join(query.sent_str) + "\t" + " ".join(value.sent_str) + "\t" + str(kernel_sim) + "\n")

    print("Normalized mean kernel-similarity: {}".format(avg_kernel_sim))


def node_height(token):
    i = 0
    while token.head != token:
        i += 1
        token = token.head
    return i


def get_tests() -> List[Dict]:
    tests = [{'func': lambda x: x.doc[x.index].dep_, 'name': 'dependency edge'},
             {'func': lambda x: x.doc[x.index].pos_, 'name': 'pos'},
             {'func': lambda x: x.doc[x.index].tag_, 'name': 'tag'},
             {'func': lambda x: x.doc[x.index].head.dep_, 'name': 'head\'s dependency edge'},
             {'func': lambda x: x.doc[x.index].i, 'name': 'index'}, 
             {'func': lambda x: gat_constituency_path_to_root(x.tree, x.index)[1:], 'name': 'constituency-path-length=until root'},
             {'func': lambda x: gat_constituency_path_to_root(x.tree, x.index)[1:4], 'name': 'constituency-path-length=3'},
             {'func': lambda x: gat_constituency_path_to_root(x.tree, x.index)[1:3], 'name': 'constituency-path-length=2'}]

    return tests


def perform_tests(query_words, k_value_words, k=1):
    tests = get_tests()
    for i in range(len(tests)):
        tests[i]['pos'] = 0.
        tests[i]['neg'] = 0.
    depth1, depth2 = [], []

    for (query, value) in zip(query_words, list(map(list, zip(*k_value_words)))):

        for t in tests:
            obj1, k_obj2 = t['func'](query), [t['func'](x) for x in value[:k]]
            if obj1 in k_obj2:
                t['pos'] += 1
            else:
                t['neg'] += 1

            if k == 1:
                depth1.append(node_height(query.doc[query.index]))
                depth2.append(node_height(value[0].doc[value[0].index]))

    for t in tests:
        acc = t['pos'] / (t['pos'] + t['neg'])
        print("Percentage of closest-words pairs with the same {0} (k={1}): {2}".format(t['name'], k, acc))
    if k == 1:
        corr, p = pearsonr(depth1, depth2)
        print("pearson correlation and p-value between the trees depth: {0}, {1}".format(corr, p))


def same_ancestor_deps(query, value):
    ancestors = set()
    token = query.doc[query.index]
    while token.dep_ != 'ROOT':
        ancestors.add(token.dep_)
        token = token.head

    dep_dic_pos = defaultdict(int)
    dep_dic_neg = defaultdict(int)

    token = value[0].doc[value[0].index]
    while token.dep_ != 'ROOT':
        if token.dep_ in ancestors:
            dep_dic_pos[token.dep_] += 1
        else:
            dep_dic_neg[token.dep_] += 1
        token = token.head
    return dep_dic_pos, dep_dic_neg


def perform_same_dep_father_test(query_words, k_value_words):
    dep_dic_pos = defaultdict(int)
    dep_dic_neg = defaultdict(int)

    for (query, value) in zip(query_words, list(map(list, zip(*k_value_words)))):
        pos_dic, neg_dic = same_ancestor_deps(query, value)
        dep_dic_pos.update(pos_dic)
        dep_dic_neg.update(neg_dic)

    for dep, pos in dep_dic_pos.items():
        acc = float(pos) / (pos + dep_dic_neg[dep])
        print("Percentage of same ancestor dep {0} : {1}".format(dep, acc))


def persist_examples(extractor, query_words, k_value_words):
    fname = "results/closest_words.extractor:{}.txt".format(extractor is not None)
    with open(fname, "w", encoding="utf8") as f:
        for (query, value) in zip(query_words, k_value_words[0]):
            query_token = query.doc[query.index]
            value_token = value.doc[value.index]
            dep1, dep2 = query_token.dep_, value_token.dep_
            correct_dep = dep1 == dep2
            word1, word2 = query_token.text, value_token.text
            sent1, sent2 = query.sentence, value.sentence
            ind1, ind2 = query_token.i, value_token.i
            sent1_str = " ".join(sent1[:ind1] + ["***" + word1 + "***"] + sent1[ind1 + 1:])
            sent2_str = " ".join(sent2[:ind2] + ["***" + word2 + "***"] + sent2[ind2 + 1:])

            f.write(sent1_str + "\t" + sent2_str + "\t" + str(dep1) + "\t" + str(dep2) + "\t" + str(correct_dep) + "\n")


def syntactic_extractor(data, extractor):
    print("Applying syntactic extractor...")
    for i, word_representation in enumerate(tqdm(data)):
        data[i] = word_representation._replace(
            word_vector=extractor.extract(word_representation.word_vector).reshape(-1))
    return data


def collect_deps_embeddings(words_reprs: List[Word_vector], extractor=None):
    if extractor is not None:

        print("Applying syntactic extractor...")

        for i, word_representation in tqdm(enumerate(data), total=len(data), ascii=True):
            words_reprs[i] = word_representation._replace(
                word_vector=extractor.extract(word_representation.word_vector).reshape(-1))

    deps = [w.doc[w.index].dep for w in words_reprs]
    counter = Counter(list)

    for vec, dep in tqdm.tqdm(zip(words_reprs, deps), ascii=True):
        counter[dep].append(vec)

    dep2vec = {}

    for dep in counter.keys():
        dep2vec[dep] = np.mean(counter[dep])

    with open("dep2vec.extractor:{}".format(extractor is not None), "wb") as f:

        pickle.dump(dep2vec, f)


def closest_word_test(words_reprs: List[Word_vector], extractor=None,
                      num_queries=15, method="cosine",
                      k=5):
    """
        Parameters
        words_reprs: A list of Word_vector, required.
        extractor: SyntacticExtractor, optional.
                   An instance of the interface SyntacticExtractor that extracts syntactic representations.
                   if None, use unmodified ELMO vectors. else, project each ELMO vectors using the extractor.
        num_queries:
                   how many closest-sentence queries to perform.
        method: str, optional (cosine / euc)
                what kind of similarity function to use.
        ----------
        """

    print("Performing closest-word test. Using extractor: {}".format(extractor is not None))

    data = copy.deepcopy(words_reprs)

    # if supplied with extractor, use it to project the vectors to the syntactic space.
    if extractor is not None:
        print("Applying syntactic extractor...")
        data = syntactic_extractor(data, extractor)

    # collect word vectors
    vecs = [word_representation.word_vector for word_representation in data]
    vecs = np.array(vecs)
    sents = [word_representation.sentence for word_representation in data]
    # perform closest_vector query.
    queries = vecs[:num_queries]
    closest_indices = get_closest_vectors(vecs, queries, sents, method=method, k=k)

    query_words = [data[i] for i in range(num_queries)]

    k_value_words = []
    for i in range(k):
        value_words = [data[closest_ind[i]] for closest_ind in closest_indices]
        k_value_words.append(value_words)

    # exact match
    perform_tests(query_words, k_value_words, k=1)

    # k nearest match
    perform_tests(query_words, k_value_words, k=k)

    perform_same_dep_father_test(query_words, k_value_words)

    # persist_examples(extractor, query_words, k_value_words)


def perform_tsne(words_reprs: List[Word_vector], extractor, num_vecs=1000, color_by="position", metric="euclidean",
                 color_by_func=color_by_dep):
    random.seed(0)
    data = random.choices(words_reprs, k=num_vecs)

    # Apply syntactic extractor

    if extractor is not None:

        print("Applying syntactic extractor...")

        for i, word_representation in tqdm(enumerate(data), total=len(data)):
            data[i] = word_representation._replace(
                word_vector=extractor.extract(word_representation.word_vector).reshape(-1)[:])

    # Collect vectors & labels

    embeddings, labels = [], []

    counter = Counter()

    for word_repr in data:
        vec = word_repr.word_vector
        embeddings.append(vec)
        label = color_by_func(word_repr)
        labels.append(label)
        counter[label] += 1

    label_set = [label for (label, count) in counter.most_common(7)]
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    if metric == "cosine":
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    # calculate TSNE projection & plot

    print("calculating projection...")

    proj = TSNE(n_components=2, random_state=0, metric=metric, verbose=1).fit_transform(embeddings)

    fig, ax = plt.subplots()

    xs, ys = proj[:, 0], proj[:, 1]
    xs, ys = list(xs), list(ys)

    if color_by == "dep":

        for label in label_set:
            condition = labels == label
            vecs = proj[condition]
            xs, ys = vecs[:, 0], vecs[:, 1]
            ax.scatter(xs, ys, label=label, alpha=0.6)
            title = 'T-SNE by Grammatical Function'

        plt.legend()

    else:

        N = len(set(label_set))
        # define the colormap
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize
        bounds = np.linspace(0, N, N + 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        scat = ax.scatter(xs, ys, c=labels, cmap=cmap, norm=norm, alpha=0.6)
        cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
        cb.set_label(color_by)

        title = "T-SNE by Position in the Sentence"

    title += "\n        (#words: {}; applied syntactic extractor: {}; metric: {})".format(num_vecs,
                                                                                          extractor is not None, metric)
    ax.set_title(title)
    plt.show()
