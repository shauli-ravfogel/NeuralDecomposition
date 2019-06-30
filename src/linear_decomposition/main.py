import argparse


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Linear Syntactic Decomposition',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-type', dest='dataset_type', type=str,
                        default='pos1gram',
                        help='Type of dataset: Bert / pos0gram / pos1gram')
    parser.add_argument('--method', dest='method', type=str,
                        default='pca',
                        help='Method: cca / pca')
    parser.add_argument('--dim', dest='dim', type=int,
                        default='64',
                        help='Number of syntactic dimensions')                        
    parser.add_argument('--reduce-first', dest='reduce', type=bool,
                        default='True',
                        help='Whether or not to perform PCA first on the entire data, to filter out noise.')  
    parser.add_argument('--reduce-dim', dest='reduce_dim', type=int,
                        default='900',
                        help='How many dimensions in the initial PCA?.') 
    
    args = parser.parse_args()
                                              
    output_filename = "decomoModel." + "" if not args.reduce else "PCA:"+str(args.reduce_dim) +".Method:" + args.method + ".Dim:" + str(args.dim) + ".pickle"
    print(output_filename)
