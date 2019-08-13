import sys

sys.path.append('../src/generate_dataset')
from utils import DEFAULT_PARAMS

FUNCTION_WORDS = DEFAULT_PARAMS["function_words"]
sys.path.insert(0, './tree_distance')

from typing import List, Tuple
import typing
from syntactic_extractor import SyntacticExtractor
import tree_similarity
import sklearn
from sklearn import metrics
import numpy as np
from tqdm.auto import tqdm
import spacy
import random

Sentence_vector = typing.NamedTuple("Sentence_vector",
                                    [('sent_vectors', np.ndarray), ('sent_str', List[str]), ("parse", List[str]),
                                     ("pos", List[str]), ("tag", List[str])])
Word_vector = typing.NamedTuple("Word_vector", [('word_vector', np.ndarray), ('sentence', List[str]), ("index", int),
                                                ("dep_edge", str), ("word", str), ("pos", str), ("tag", str)])


def run_tests(embds_and_sents: List[Tuple[List[np.ndarray], str]], extractor, num_queries, method, num_words,
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

    sentence_reprs = get_sentence_representations(embds_and_sents)  # sentence representatons
    words_reprs = sentences2words(sentence_reprs, num_words=num_words,
                                  ignore_function_words=ignore_function_words)  # words representatons

    # closest-word, with ELMO alone (basline)
    closest_word_test(words_reprs, num_queries=num_queries, method=method, extractor=None)

    # closest-word, with ELMO + syntactic extractor
    closest_word_test(words_reprs, num_queries=num_queries, method=method, extractor=extractor)

    # closest-sentence, with ELMO alone (basline)
    closest_sentence_test(sentence_reprs, num_queries=num_queries, method=method, extractor=None)

    # closest-sentence, with ELMO + syntactic extractor
    closest_sentence_test(sentence_reprs, num_queries=num_queries, method=method, extractor=extractor)


def parse(sentences: List[List[str]]) -> (List[List[str]], List[List[str]], List[List[str]]):
    """
        Parameters
        
        sentences: A list of sentence, where each sentence is a list of word strings.
        ----------
        Returns
        -------
        all_deps: A list of dependency edges, where all_deps[i][j] is the dep of word j in sentence i.
        """

    print("Parsing...")

    tokens_dict = {" ".join(sentence): sentence for sentence in sentences}

    def custom_tokenizer(text):
        return tokens_dict[text]

    nlp = spacy.load('en_core_web_sm')

    all_deps = []
    all_pos = []
    all_tags = []
    count = 0

    for sent in tqdm(sentences, ascii=True):

        doc = spacy.tokens.Doc(vocab=nlp.vocab, words=sent)
        for name, proc in nlp.pipeline:
            doc = proc(doc)

        deps = [token.dep_ for token in doc]
        all_deps.append(deps)

        pos = [token.pos_ for token in doc]
        all_pos.append(pos)

        tags = [token.tag_ for token in doc]
        all_tags.append(tags)

        assert len(deps) == len(sent) == len(pos) == len(tags)

    return all_deps, all_pos, all_tags


def get_closest_vectors(all_vecs: List[np.ndarray], queries: List[np.ndarray], method: str, k=5):
    if method == "cosine":

        # normalzie the vectors

        all_vecs = all_vecs / np.linalg.norm(all_vecs, axis=1)[:, None]
        queries = queries / np.linalg.norm(queries, axis=1)[:, None]

        # perform dot product

        distances = sklearn.metrics.pairwise_distances(queries, all_vecs, metric="cosine")

    else:
        distances = sklearn.metrics.pairwise_distances(queries, all_vecs, metric="euclidean")

    top_k = distances.argsort(axis=1)[:, :k]
    closest_indices = top_k[:, 1]  # ignore the word itself

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
    deps, pos, tags = parse(sentences)

    assert len(deps) == len(sentences) == len(embds) == len(pos) == len(tags)

    embds_sents_deps = [Sentence_vector(e, s, d, p, t) for e, s, d, p, t in zip(embds, sentences, deps, pos, tags)]

    return embds_sents_deps


def sentences2words(sentence_representations: List[Sentence_vector], num_words, ignore_function_words=True) -> List[
    Word_vector]:
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

        vectors, words, deps, pos, tags = sent_rep

        for j, (vec, w, dep, p, t) in enumerate(zip(vectors, words, deps, pos, tags)):

            if ignore_function_words and w in FUNCTION_WORDS: continue

            data.append(Word_vector(vec.copy(), words, j, dep, w, p, t))

    random.seed(0)
    random.shuffle(data)

    print("Collected {} words".format(len(data)))
    return data


def closest_sentence_test(sentence_representations: List[Sentence_vector], extractor=None, num_queries=15,
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
    closest_indices = get_closest_vectors(vecs, queries, method=method)

    query_sents = [sents[i] for i in range(num_queries)]
    value_sents = [sents[closest_ind] for closest_ind in closest_indices]

    kernel_sims, edit_sims = tree_similarity.get_similarity_scores(query_sents, value_sents)
    avg_kernel_sim = np.mean(kernel_sims)
    avg_edit_sims = np.mean(edit_sims)

    fname = "results/closest_sentences.extractor:{}.txt".format(extractor is not None)
    with open(fname, "w", encoding="utf8") as f:

        for (query, value, kernel_sim, edit_sim) in zip(query_sents, value_sents, kernel_sims, edit_sims):
            f.write(" ".join(query) + "\t" + " ".join(value) + "\t" + str(kernel_sim) + "\t" + str(edit_sim) + "\n")

    print("Normalized mean kernel-similarity: {}; Normalized mean edit-similarity: {}".format(avg_kernel_sim,
                                                                                              avg_edit_sims))


def closest_word_test(words_reprs: List[Word_vector], extractor=None, num_queries=15, method="cosine"):
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

    data = words_reprs

    # if supplied with extractor, use it to project the vectors to the syntactic space.
    # (TODO: should probably be moved to a separate function. [sentences2words?])

    if extractor is not None:

        print("Applying syntactic extractor...")

        for i, word_representation in tqdm(enumerate(data), total=len(data)):
            data[i] = word_representation._replace(
                word_vector=extractor.extract(word_representation.word_vector).reshape(-1))

    # collect word vectors

    vecs = [word_representation.word_vector for word_representation in data]
    vecs = np.array(vecs)

    # perform closest_vector query.

    queries = vecs[:num_queries]
    closest_indices = get_closest_vectors(vecs, queries, method=method)

    query_words = [data[i] for i in range(num_queries)]
    value_words = [data[closest_ind] for closest_ind in closest_indices]

    good_dep, bad_dep = 0., 0.
    good_pos, bad_pos = 0., 0.
    good_tag, bad_tag = 0., 0.

    fname = "results/closest_words.extractor:{}.txt".format(extractor is not None)
    with open(fname, "w", encoding="utf8") as f:

        for (query, value) in zip(query_words, value_words):

            dep1, dep2 = query.dep_edge, value.dep_edge
            pos1, pos2 = query.pos, value.pos
            tag1, tag2 = query.tag, value.tag
            correct_dep = dep1 == dep2
            word1, word2 = query.word, value.word
            sent1, sent2 = query.sentence, value.sentence
            ind1, ind2 = query.index, value.index
            sent1_str = " ".join(sent1[:ind1] + ["***" + word1 + "***"] + sent1[ind1 + 1:])
            sent2_str = " ".join(sent2[:ind2] + ["***" + word2 + "***"] + sent2[ind2 + 1:])

            f.write(sent1_str + "\t" + sent2_str + "\t" + str(dep1) + "\t" + str(dep2) + "\t" + str(correct_dep) + "\n")

            if correct_dep:
                good_dep += 1
            else:
                bad_dep += 1

            if pos1 == pos2:
                good_pos += 1
            else:
                bad_pos += 1

            if tag1 == tag2:
                good_tag += 1
            else:
                bad_tag += 1

    acc = good_dep / (good_dep + bad_dep)
    print("Percentage of closest-words pairs with the same dependency-edge: {}".format(acc))
    print("Percentage of closest-words pairs with the same pos: {}".format(good_pos / (good_pos + bad_pos)))
    print("Percentage of closest-words pairs with the same tag: {}".format(good_tag / (good_tag + bad_tag)))


def perform_tsne(words_reprs: List[Word_vector], extractor, num_vecs=1000, color_by="position", metric="euclidean"):
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

    dep_counter = Counter()
    position_counter = Counter()

    for word_repr in data:
        vec, dep, ind = word_repr.word_vector, word_repr.dep_edge, word_repr.index
        embeddings.append(vec)
        labels.append(dep if color_by == "dep" else ind)
        dep_counter[dep] += 1
        position_counter[ind] += 1

    counter = position_counter if color_by == "position" else dep_counter
    label_set = [label for (label, count) in counter.most_common(18)]
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    if metric == "cosine":
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    # calculate TSNE projection & plot

    print("calculating projection...")

    proj = TSNE(n_components=2, random_state=0, metric=metric).fit_transform(embeddings)

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

    elif color_by == "position":

        N = len(set(labels))
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
        cb.set_label('Position')

        title = "T-SNE by Position in the Sentence"

    title += "\n        (#words: {}; applied syntactic extractor: {}; metric: {})".format(num_vecs,
                                                                                          extractor is not None, metric)
    ax.set_title(title)
    plt.show()
