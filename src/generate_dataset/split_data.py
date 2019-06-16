import argparse
import random
from sklearn.model_selection import train_test_split


random.seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--filename', dest='file_name', type=str,
                        default='../data/interim/data.txt',
                        help='name of the source text file')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default='../data/processed/',
                        help='output directory, where to put the train/dev/test splits')
    parser.add_argument('--train-size', dest='train_size', type=float, default=0.8,
                        help='portion of the training size')

    args = parser.parse_args()

    with open(args.file_name, "r") as f:
        data = f.readlines()

    train, eval_split = train_test_split(data, test_size=(1.0 - args.train_size), random_state=42)
    dev, test = train_test_split(eval_split, test_size=0.5, random_state=42)

    parts = [train, dev, test]

    for part, name in zip(parts, ["train", "dev", "test"]):

        with open(args.output_dir + name, "w") as f:
            for line in part:
                f.write(line)
