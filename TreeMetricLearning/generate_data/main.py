import argparse
import model
from model_runner import ModelRunner
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-wiki', dest='input_wiki', type=str,
                        default='../data/external/wikipedia.sample.tokenized',
                        help='name of the source wikipedia text file')
    parser.add_argument('--w2v-file', dest='w2v_file', type=str,
                        default='data/external/GoogleNews-vectors-negative300.bin',
                        help='name of the source wikipedia text file')
    parser.add_argument('--output-data', dest='output_data', type=str,
                        default='../data/interim/data.txt',
                        help='name of the output file')
    parser.add_argument('--output-sentences', dest='output_sentences', type=str,
                        default='data/interim/emb_sents.pickle',
                        help='name of the output file')
    parser.add_argument('--pos2words-file', dest='pos2words_file', type=str,
                        default='data/external/pos2words.pickle',
                        help='name of the output file')
    parser.add_argument('--num-sentences', dest='num_sentences', type=int, default=20,
                        help='Number of equivalent sentences to generate from each sentence.')
    parser.add_argument('--substitutions-type', dest='substitution_type', type=str,
                        default='embeddings')
    parser.add_argument('--substitutions-file', dest='substitution_file', type=str,
                        default='')
    parser.add_argument('--elmo_folder', dest='elmo_folder', type=str,
                        default='../data/external')
    parser.add_argument('--cuda-device', dest='cuda_device', type=int, default=0,
                        help='cuda device to run the LM on')

    args = parser.parse_args()

    elmo_folder = args.elmo_folder

    model = model.Elmo(elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_options.json',
                       elmo_folder + '/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
                       args.cuda_device)
    model_runner = ModelRunner(model, args.input_wiki, args.output_data, persist=True)
    model_runner.run()
