import evaluator
import syntactic_extractor
import args


if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--extractor', dest='extractor_type', type=str,
                        default='cca',
                        help='pca / cca')
        
        args = parser.parse()
        
        if args.extractor_type == "cca":
        
                        
                extractor = syntactic_extractor.CCASyntacticExtractor()
        else:
                extractor = syntactic_extractor.CCASyntacticExtractor()
                
        evaluator = evaluator.Evaluator(extractor)
        evaluator.test()
