import pickle
import gensim
import numpy as np
from typing import DefaultDict, List, Tuple, Dict, Set
import sklearn
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances
import tqdm
import random
from utils import DEFAULT_PARAMS

FUNCTION_WORDS = DEFAULT_PARAMS["function_words"]


def load_embeddings(w2v_file: str):
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    word_set = set(list(embeddings.wv.vocab))

    return embeddings, word_set


def load_sents(path: str) -> List[List[List[str]]]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    return list(data.values())


def calcualte_similarity_score(sent_vecs: np.ndarray, group_function_mask: np.ndarray, ignore_function=True) -> float:
    relevant = sent_vecs * group_function_mask[..., None] if ignore_function else sent_vecs.copy()

    bag_of_words = np.sum(relevant, axis=1)  # sum over seq length dimension.
    dis_mat = cosine_distances(bag_of_words)
    mean_distance = np.mean(dis_mat.flatten())
    return mean_distance


def sort_by_similarity_score(groups: List[List[List[str]]], embds: gensim.models.keyedvectors.Word2VecKeyedVectors,
                             vocab: Set[str]):
    scores = np.zeros(len(groups))

    for i, group in tqdm.tqdm(enumerate(groups), ascii=True, total=len(groups)):

        group_size, sent_len = group.shape
        group_vecs = np.zeros((group_size, sent_len, 300))
        group_function_mask = np.zeros((group_size, sent_len))

        for j, sent in enumerate(group):
            for k, w in enumerate(sent):
                group_vecs[j, k] = embds[w] if w in vocab else embds["##"]
                group_function_mask[j, k] = 1. if w not in FUNCTION_WORDS else 0.

        sim_score = calcualte_similarity_score(group_vecs, group_function_mask)
        scores[i] = sim_score

    groups_and_sim_scores = list(zip(groups, scores))
    groups_and_sim_scores = sorted(groups_and_sim_scores, key=lambda group_and_score: group_and_score[1])

    with open("groups_and_scores", "wb") as f:

        pickle.dump(groups_and_sim_scores, f)

    return groups_and_sim_scores


def print_group(group):
    for sent in group:
        print(" ".join(sent))
        print("----------------------")
    print("========================================")


def print_sents_by_percentile(sorted_sents: List[Tuple[np.ndarray, float]]):
    step = 0.1
    percentiles = step + np.arange(0, 1, step)
    l = len(sorted_sents)
    examples_per_group = 10

    for per in percentiles:

        low, top = int(l * (per - step)), int(l * per)
        relevant = sorted_sents[low:top]

        print(per, low, top)
        for k in range(examples_per_group):
            random_sent, score = random.choice(relevant)
            print("score: {}".format(score))
            print_group(random_sent)


def main():
    embds, vocab = load_embeddings("../../data/external/GoogleNews-vectors-negative300.bin")
    groups = load_sents("../../data/interim/bert_online_sents_same_pos5.pickle")
    groups_and_sim_scores = sort_by_similarity_score(groups, embds, vocab)
    print_sents_by_percentile(groups_and_sim_scores)


if __name__ == '__main__':
    main()
