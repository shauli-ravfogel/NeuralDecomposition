import argparse
import pca_decomp
import cca_decomp

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Linear Syntactic Decomposition',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-type', dest='dataset_type', type=str,
                        default='pos1gram',
                        help='Type of dataset: Bert / pos0gram / pos1gram')
    parser.add_argument('--method', dest='method', type=str,
                        default='cca',
                        help='Method: cca / pca')
    parser.add_argument('--syn-dim', dest='syntactic_dim', type=int,
                        default=100,
                        help='Number of syntactic dimensions')  
    parser.add_argument('--semantic_dim', dest='sem_dim', type=int,
                        default=200,
                        help='Number of semantic dimensions (only relevant for PCA)')                       
    parser.add_argument('--reduce-first', dest='reduce', type=bool,
                        default='True',
                        help='Whether or not to perform PCA first on the entire data, to filter out noise.')  
    parser.add_argument('--reduce-dim', dest='reduce_dim', type=int,
                        default=900,
                        help='How many dimensions in the initial PCA?.') 
    parser.add_argument('--num_sentences', dest='num_sentences', type=int,
                        default=15,
                        help='How many sentences?.')     
    args = parser.parse_args()
    
    data = "../data/interim/bert_online_data.txt"                                              
    output_filename = "decomoModel." + "" if not args.reduce else "PCA:"+str(args.reduce_dim) +".Method:" + args.method + ".Dim:" + str(args.syntactic_dim) + ".Data:" +args.dataset_type + ".Sentences:" + str(args.num_sentences) +  ".pickle"
    
    if args.method == "cca":
    
        decomp = cca_decomp.CCADecomposition(args.syntactic_dim, args.reduce, args.reduce_dim, args.num_sentences, data)
