import argparse


def read_file(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def write_file(output_file, data):
    with open(output_file, 'w') as f:
        for line in data:
            f.write(line + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='filter wikipedia sentences',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-wiki', dest='input_wiki', type=str,
                        default='data/external/wiki.clean.random',
                        help='name of the source wikipedia text file')
    parser.add_argument('--min-length', dest='min_length', type=int, default=10,
                        help='minimum sentence length to filter')
    parser.add_argument('--max-length', dest='max_length', type=int, default=30,
                        help='maximum sentence length to filter')
    parser.add_argument('--train-size', dest='train_size', type=int, default=250000,
                        help='filter training part of wikipedia')
    parser.add_argument('--out-file', dest='out_file', type=str,
                        default='data/external/wiki.clean.eval.txt',
                        help='output file name of wikipedia data after filtering')

    args = parser.parse_args()

    data = read_file(args.input_wiki)
    filtered_data = list(filter(lambda x: args.min_length < len(x.split()) < args.max_length, data))

    eval_data = filtered_data[args.train_size:]
    write_file(args.out_file, eval_data)
