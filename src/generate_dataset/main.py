import generators
import argparse
import model
from model_runner import ModelRunner, TuplesModelRunner
import pickle

#import torch.backends
#torch.backends.cudnn.benchmark=True
#torch.backends.cudnn.fastest=True

pos_tags_to_replace = ["NN", "NNS", "NNP", "NNPS", "PRP$", "JJ", "CD", "VB", "VBD", "VBG", "VBN",
                       "VBP", "VBZ"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-wiki', dest='input_wiki', type=str,
                        default='../../data/external/wiki.clean.250k',
                        help='name of the source wikipedia text file')
    parser.add_argument('--w2v-file', dest='w2v_file', type=str,
                        default='../../data/external/GoogleNews-vectors-negative300.bin',
                        help='name of the source wikipedia text file')
    parser.add_argument('--output-data', dest='output_data', type=str,
                        default='../../data/interim/elmo_states.fwd.wiki.hdf5',
                        help='name of the output file')
    parser.add_argument('--output-sentences', dest='output_sentences', type=str,
                        default='../../data/interim/bert.wiki.repeat=1.guesses=30.pickle',
                        help='name of the output file')
    parser.add_argument('--pos-tags-to-replace', dest='pos_tags_to_replace', type=list,
                        default=pos_tags_to_replace, help='which POS tags to replace')
    parser.add_argument('--pos2words-file', dest='pos2words_file', type=str,
                        default='../../data/external/pos2words.pickle',
                        help='name of the output file')
    parser.add_argument('--num-sentences', dest='num_sentences', type=int, default=7,
                        help='Number of equivalent sentences to generate from each sentence.')
    parser.add_argument('--substitutions-type', dest='substitution_type', type=str,
                        default='bert')
    parser.add_argument('--substitutions-file', dest='substitution_file', type=str,
                        default = '../../data/interim/bert.wiki.repeat=1.guesses=30.pickle')
    parser.add_argument('--elmo_folder', dest='elmo_folder', type=str,
                        default='../../data/external')
    parser.add_argument('--cuda-device', dest='cuda_device', type=int, default=2,
                        help='cuda device to run the LM on')
    parser.add_argument('--dataset-type', dest='dataset_type', type=str, default="all",
                        help='all / pairs')
    parser.add_argument('--layers', '--list', dest = "layers", help='list of ELMO layers to include', type=str, default = "0,1,2")


    args = parser.parse_args()

    layers = [int(item) if item.isdigit() else item for item in args.layers.split(',')]

    if args.layers != "mean":
        layers = [int(item) for item in args.layers.split(',')]
    else:
        layers = "mean"

    maintain_pos = True

    # If no substitution file is provided, need to build these
    if args.substitution_file == '':

        # Reading the type of substitution technique
        if args.substitution_type == 'embeddings':
            generator = generators.EmbeddingBasedGenerator(args.input_wiki, args.output_sentences,
                                                           args.num_sentences,
                                                           args.w2v_file, 7)
        elif args.substitution_type == 'pos':
            generator = generators.POSBasedEGenerator2(args.input_wiki, args.output_sentences,
                                                      args.pos_tags_to_replace, args.num_sentences,
                                                      args.pos2words_file)
        else:
            #generator = generators.OnlineBertGenerator(args.input_wiki, args.output_sentences,
            #                                          args.num_sentences)
            generator = generators.BatchedOnlineBertGenerator(args.input_wiki, args.output_sentences,
                                                      args.num_sentences, topn = (25 if maintain_pos else 20), ignore_first_k = 0, maintain_pos = maintain_pos, cuda_device = args.cuda_device, num_repeats = 1) # was 13

        equivalent_sentences = generator.generate()
    # otherwise, reading that file
    else:
        with open(args.substitution_file, "rb") as f:
            equivalent_sentences = pickle.load(f)

    use_elmo = True

    if use_elmo:
        elmo_folder = args.elmo_folder

        model = model.Elmo(elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                       elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
                       args.cuda_device, layers, only_fwd = True)
    else:
        raise NotImplementedError('need to chose of the available random states')

    USE_ELMO = True
    if not USE_ELMO:
      print("Using BERT")
      model = model.Bert(args.cuda_device, layers = layers)

    if args.dataset_type == "pairs":
        model_runner = TuplesModelRunner(model, equivalent_sentences, args.output_data, persist=True)
    else:
        model_runner = ModelRunner(model, equivalent_sentences, args.output_data, persist=True)
    model_runner.run()
