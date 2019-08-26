import embedder
import pickle
import evaluate
import argparse
import syntactic_extractor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-wiki', dest='input_wiki', type=str,
                        default='../data/external/wiki.clean.250k',
                        help='name of the source wikipedia text file')
    parser.add_argument('--encode_sentences', dest='encode_sentences', type=bool,
                        default=False,
                        help='whether to encode the sentences with ELMO from scratch or use already calcuated '
                             'representation')
    parser.add_argument('--encoded_data', dest='encoded_data', type=str,
                        default='../data/interim/encoded_sents.pickle',
                        help='path to the embedded sentences data')
    parser.add_argument('--elmo_folder', dest='elmo_folder', type=str,
                        default='../data/external')
    parser.add_argument('--method', dest='method', type=str,
                        default='euc', help="similarity method (cosine / euc)")
    parser.add_argument('--cuda-device', dest='cuda_device', type=int, default=0,
                        help='cuda device to run the LM on')
    parser.add_argument('--num_sents', dest='num_sents', type=int, default=25000,
                        help='number of wiki sentences to use')
    parser.add_argument('--num_words', dest='num_words', type=int, default=100000,
                        help='number of total words to collect')
    parser.add_argument('--num_queries', dest='num_queries', type=int, default=5000,
                        help='number of closest-vector queries to perform within the tests.')
    parser.add_argument('--extractor', dest='extractor', type=str, default="cca",
                        help='type of syntactic extracor (cca / neural_cca)')
    parser.add_argument('--extractor_path', dest='extractor_path', type=str, default="../src/linear_decomposition/models/...",
                        help='path to the fitted extractor model')
    args = parser.parse_args()

    # use already-collected representations
    if (not args.encode_sentences) and args.encoded_data != '':
        with open(args.encoded_data, "rb") as f:
            data = pickle.load(f)

    # recalculate representations
    else:
        elmo_embedder = embedder.Embedder(args.elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                                          args.elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
                                          args.input_wiki, args.num_sents,
                                          device=args.cuda_device)
        data = elmo_embedder.get_data()

        with open(args.encoded_data, "wb") as f:
            pickle.dump(data, f)

    extractor = syntactic_extractor.CCASyntacticExtractor(args.extractor_path)

    # Run tests.
    evaluate.run_tests(data, extractor, num_queries=args.num_queries, method=args.method,
                       num_words=args.num_words, ignore_function_words=True)
