DEFAULT_PARAMS = {"file_name": "resources/wikipedia.sample.tokenized",
                  "pos2words_filename": "resources/pos2words.pickle",
                  "sentences_dict_filename": "resources/same_pos/sents.pickle",
                  "dataset_filename": "resources/same_pos/data.txt",
                  "elmo_options": "resources/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                  "elmo_weights": "resources/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                  "word2vec": "resources/GoogleNews-vectors-negative300.bin",
                  "num_sentences": 30,
                  "pos_tags_to_replace": ["NN", "NNS", "NNP", "NNPS", "PRP$", "JJ", "CD", "VB", "VBD", "VBG", "VBN",
                                          "VBP", "VBZ"],
                  "function_words": set(["in", "on", "at", "the", "a", "an", "who", "that", "this", "which", "can",
                                     "cannot", "not", "no", "do", "does", "of", "these", "those", ",", ".", "'", "(",
                                     ")", "under", "above", "near", "without", "with", "have", "having", "has", "as", "nor", "am",
                                     "is", "are", "although", "while", "between", "neither", "and", "or",
                                     "among", "anything", "beside", "besides", "both", "off", "did", "nothing", "now",
                                     "over", "rather", "perhaps", "sometime", "such", "upon",
                                     "whereas", "where", "when", "who", "whom", "what", "why", "yes", "yet", "were", "was", "been", "be", "will", "would","could", "should", "amongst", "always", "along", "all", "afterwards", "after", "'s", "during", '"', "for", "from", "to", "into", "there", "instead", "-", ":", "-", ";", "?", "about", "but", "something", "out", "up", "it", "being", "just", "i", "'ve", "some", "against", "...", "'re", "much", "``", "''", "only", "least", "first", "n't", "its", "'ll", "--", "more", "such", "how", "by", "thus", "[", "]", "/", "sometime", "sometimes", "so", "even", "got", "gotten", "get", "due", "since", "because", "though", "however", "why", "off", "one", "very", "if", "until", "then", "than", "must", "through", "almost", "any", "may", "further", "less", "least", "worthy", "course", "before", "beforehand", "either", "whatever", "behalf", "well", "had", "need", "ought", "whether", "own", "according", "accordingly", "regarding", "you", "he", "mine", "our", "his", "her", "she", "my", "they", "their", "most", "!", "?", "each", "too", "once", "again", "soon", "apart", "enough", "few", "many", "forth", "thereafter", "several", "times", "ever", "simply", "specific", "per", "underneath", "beneath", "every", "er", "ed", "ing", "whole", "alone", "nearby", "within", "whom", "toward", "towards", "doesn't", "dont", "don't", "doesnt", "probably", "same", "other", "we", "nevertheless", "via", "already", "various", "still", "aftermath", "despite", "none", "i", "ii", "beyond"])
                  }


def read_sentences(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    splitted = [line.strip().split(" ") for line in lines]

    return splitted


def to_string(np_array):
    shape = np_array.shape

    return " ".join(["%0.4f" % x for x in np_array])
