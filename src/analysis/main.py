from embedder import EmbedElmo, EmbedBert, EmbedRandomElmo
import pickle
import evaluate
import argparse
import syntactic_extractor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-wiki', dest='input_wiki', type=str,
                        default='data/external/wiki.clean.1million.eval',
                        help='name of the source wikipedia text file')
    parser.add_argument('--encode_sentences', dest='encode_sentences', type=bool,
                        default=False,
                        help='whether to encode the sentences with ELMO from scratch or use already calcuated '
                             'representation')
    parser.add_argument('--encoded_data', dest='encoded_data', type=str,
                        default='data/interim/encoded_sents.pickle',
                        help='path to the embedded sentences data')
    parser.add_argument('--elmo_folder', dest='elmo_folder', type=str,
                        default='data/external')
    parser.add_argument('--method', dest='method', type=str,
                        default='cosine', help="similarity method (cosine / euc)")
    parser.add_argument('--cuda-device', dest='cuda_device', type=int, default=0,
                        help='cuda device to run the LM on')
    parser.add_argument('--num_sents', dest='num_sents', type=int, default=35000,
                        help='number of wiki sentences to use')
    parser.add_argument('--num_words', dest='num_words', type=int, default=150000,
                        help='number of total words to collect')
    parser.add_argument('--num_queries', dest='num_queries', type=int, default=4000,
                        help='number of closest-vector queries to perform within the tests.')
    parser.add_argument('--extractor', dest='extractor', type=str, default="cca",
                        help='type of syntactic extracor (cca / neural_cca / numpy_cca / triplet)')
    parser.add_argument('--extractor_path', dest='extractor_path', type=str,
                        default="src/linear_decomposition/models/..", help='path to the fitted extractor model')
    parser.add_argument('--embedder_type', dest='embedder_type', type=str,
                        default="elmo", help='elmo / elmo_rand_lstm / elmo_rand_all / bert')
    parser.add_argument('--layers', '--list', dest = "layers", help='list of bert/elmo layers to include', type=str, default = "16,mean")

    args = parser.parse_args()
    layers = [int(item) if item.isdigit() else item for item in args.layers.split(',')]
    
    # use already-collected representations
    if (not args.encode_sentences) and args.encoded_data != '':
        with open(args.encoded_data, "rb") as f:
            sentence_reprs = pickle.load(f)

    # recalculate representations
    else:
        embedder_type = args.embedder_type
        options = {'elmo_options_path': args.elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                   'elmo_weights_path': args.elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'}
        if embedder_type == 'elmo':
            embedder = EmbedElmo(options, device=args.cuda_device)
        elif embedder_type == 'elmo_rand_lstm':
            embedder = EmbedRandomElmo(options, device=args.cuda_device, random_emb=False, random_lstm=True)
        elif embedder_type == 'elmo_rand_all':
            embedder = EmbedRandomElmo(options, device=args.cuda_device, random_emb=True, random_lstm=True)
        else:
            embedder = EmbedBert({}, device=args.cuda_device, layers = layers)
        data = embedder.get_data(args.input_wiki, args.num_sents)
        sentence_reprs = evaluate.get_sentence_representations(data)
        
        with open(args.encoded_data, "wb") as f:
            pickle.dump(sentence_reprs, f)

    if args.extractor == "cca":
        extractor = syntactic_extractor.CCASyntacticExtractor(args.extractor_path, numpy=False)
    elif args.extractor == "numpy_cca":
        print(args.extractor_path)
        extractor = syntactic_extractor.CCASyntacticExtractor(args.extractor_path, numpy=True)
    elif args.extractor == "triplet":
            extractor = syntactic_extractor.TripletLossModelExtractor(args.extractor_path)
    else:
        raise NotImplementedError()
    # Run tests.
    evaluate.run_tests(sentence_reprs, extractor, num_queries=args.num_queries, method=args.method,
                       num_words=args.num_words, ignore_function_words=True)
