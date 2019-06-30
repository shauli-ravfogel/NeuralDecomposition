import evaluator
import syntactic_extractor


if __name__ == '__main__':

        extractor = syntactic_extractor.CCASyntacticExtractor()
        evaluator = evaluator.Evaluator(extractor)
