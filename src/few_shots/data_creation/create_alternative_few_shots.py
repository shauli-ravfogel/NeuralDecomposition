"""
python create_few_shot.py --input-file /home/lazary/workspace/thesis/NeuralDecomposition/data/parsing/train

in_f /home/nlp/ravfogs/neural_decomposition/data/interim/penn_sents.pickle
"""
import argparse
import os
import numpy as np
import pickle
from typing import List
import nltk
import copy


def add_parentheses(sent: List[str], tree: nltk.tree.Tree):
    """
    adds parentheses to the input sentence, according to the known tree sturcutre.
    """

    for (leaf_position, w) in zip(tree.treepositions('leaves'), sent):
        tree[leaf_position] = w

    parse_string = ' '.join(str(tree).split())
    return parse_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-file', dest='input_file', type=str,
                        help='name of training data file of the penn treebank data')
    parser.add_argument('--origin-file', dest='origin_file', type=str,
                        help='name of training data file of the penn treebank data')
    parser.add_argument('--split', dest='split', type=str,
                        help='split part (train/dev/test)')

    args = parser.parse_args()

    input_file = args.input_file
    origin_file = args.origin_file
    input_dir = os.path.dirname(input_file)
    split = args.split

    data_size = map(int, args.data_size.split(','))

    with open(input_file, "rb") as f:
        penn_data = pickle.load(f)
    split_data = penn_data[split]

    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    np.random.seed(16)
    alternative_data = []
    for suggestion, tree in zip(split_data, lines):
        original_tree = nltk.tree.Tree.fromstring(tree)
        ind = np.random.randint(1, 6)
        suggestion_tree = add_parentheses(suggestion[ind], copy.deepcopy(original_tree))

    np.random.seed(16)
    np.random.shuffle(lines)

    for split in alternative_data:
        print(len(split))
        with open(os.path.join(input_dir, split + '_' + str(len(split))), 'w') as f:
            for line in split:
                f.write(line + '\n')
