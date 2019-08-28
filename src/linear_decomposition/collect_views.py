import argparse
import views_collector

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Views collection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-path', dest='input_path', type=str,
                        default='data/interim/encoder_bert/sents_bert_base.hdf5',
                        help='name of the hdf5 input file (containing encoded equivalent sentences)')
    parser.add_argument('--output-path', dest='output_path', type=str,
                        default='data/interim/views/',
                        help='directory where to write the views')
    parser.add_argument('--num_examples', dest='num_examples', type=int,
                        default=2000000,
                        help='how many pairs to collect')
    parser.add_argument('--mode', dest='mode', type=str,
                        default="simple",
                        help='simple / averaged / sentence-level')
    parser.add_argument('--exclude_function_words', dest='exclude_function_words', type=bool,
                        default=True,
                        help='whether or not to exclude function words from the pairs')
    parser.add_argument('--output-file', dest='output_file', type=str,
                        help='extra string to add to the output file')

    args = parser.parse_args()

    collector_args = (args.input_path, args.output_path,
                      args.num_examples, args.mode, args.exclude_function_words)

    if args.mode == "simple":
        collector = views_collector.SimpleCollector(*collector_args)
    elif args.mode == "averaged":
        collector = views_collector.AveragedCollector(*collector_args)
    elif args.mode == "sentence-level":
        collector = views_collector.SentenceCollector(*collector_args)

    collector.collect_views(args.output_file)
