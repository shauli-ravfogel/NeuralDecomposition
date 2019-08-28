import argparse
import cca
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Views collection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--views-file-path', dest='views_path', type=str,
                        default='views/views.sentences:17845.pairs:2500120.mode:simple.no-func-words:True',
                        help='name of the views file')
    parser.add_argument('--perform-pca', dest='perform_pca', type=bool,
                        default=False,
                        help='whether or not to perform PCA')
    parser.add_argument('--pca-dim', dest='pca_dim', type=int,
                        default=1950,
                        help='if perform_pca, PCA dimensionality')
    parser.add_argument('--cca-dim', dest='cca_dim', type=int,
                        default=100,
                        help='CCA dimensionality')
    parser.add_argument('--enforce-symmetry', dest='enforce_symmetry', type=bool,
                        default=True,
                        help='whether to enforce symmetry on the CCA matrices (by adding an example (y,x) for each example (x,y))')
    parser.add_argument('--cca-model', dest='model', type=str,
                        default="numpy",
                        help='numpy / sklearn. whether to use sklearn CCA or vanilla numpy implemenetation)')
    parser.add_argument('--whiten', dest='whiten', type=bool,
                        default=False,
                        help='Whether to perform whitening')
    parser.add_argument('--num_examples', dest='num_examples', type=int,
                        default = 1000000,
                        help='num_examples')
    parser.add_argument('--output-path', dest='output_path', type=str,
                        default="data/processed/models/",
                        help='directory where to store the model')

    args = parser.parse_args()

    cca_model = cca.run_cca(args.views_path, args.perform_pca, args.pca_dim, args.cca_dim, args.enforce_symmetry,
                            args.model, args.whiten, args.num_examples)

    filename = args.output_path + "/cca.perform-pca:{}.cca-dim:{}.symmetry:{}.whitening:{}.examples:{}.method:{}.pickle".format(args.perform_pca,
                                                                                             args.cca_dim,
                                                                                             args.enforce_symmetry, args.whiten, args.num_examples, args.model)

    with open(filename, "wb") as f:
        pickle.dump(cca_model, f)
