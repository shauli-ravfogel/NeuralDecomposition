"""
python create_few_shot.py --input-train /home/lazary/workspace/thesis/NeuralDecomposition/data/parsing/train
"""
import argparse
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-train', dest='input_train', type=str,
                        help='name of training data file of the penn treebank data')
    parser.add_argument('--data-size', dest='data_size', type=str, default='50,100,200,500,1000',
                        help='different sizes split of the training set')

    args = parser.parse_args()

    input_file = args.input_train
    input_dir = os.path.dirname(input_file)

    data_size = map(int, args.data_size.split(','))

    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    np.random.seed(16)
    np.random.shuffle(lines)

    splits = []
    for size in data_size:
        splits.append(lines[:size])
        print(size)

    for split in splits:
        print(len(split))
        with open(os.path.join(input_dir, 'train_' + str(len(split))), 'w') as f:
            for line in split:
                f.write(line + '\n')
