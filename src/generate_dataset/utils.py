DEFAULT_PARAMS = {"file_name": "resources/wikipedia.sample.tokenized",
                  "pos2words_filename": "resources/pos2words.pickle",
                  "sentences_dict_filename": "resources/same_pos/sents.pickle",
                  "dataset_filename": "resources/same_pos/data.txt",
                  "elmo_options": "resources/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                  "elmo_weights": "resources/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                  "word2vec": "resources/GoogleNews-vectors-negative300.bin",
                  "num_sentences": 20,
                  "pos_tags_to_replace": ["NN", "NNS", "NNP", "NNPS", "PRP$", "JJ", "CD", "VB", "VBD", "VBG", "VBN",
                                          "VBP", "VBZ"],
                  "function_words": ["in", "on", "at", "the", "a", "an", "who", "that", "this", "which", "can",
                                     "cannot", "not", "no", "do", "does", "of", "these", "those", ",", ".", "'", "(",
                                     ")", "under", "above", "near", "without", "have", "has", "got", "as", "nor", "am",
                                     "is", "are", "although", "while", "between", "neither", "and", "or", "one",
                                     "among", "anything", "beside", "besides", "both", "off", "did", "nothing", "now",
                                     "over", "rather", "perhaps", "sometime", "such", "still", "upon", "up", "down",
                                     "whereas", "where", "why", "yes", "yet", "were", "was", "been", "be", "will",
                                     "would", "amongst", "always", "along", "all", "afterwards"]
                  }


def read_sentences(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

    splitted = [line.strip().split(" ") for line in lines]

    return splitted


def to_string(np_array):
    shape = np_array.shape

    return " ".join(["%0.4f" % x for x in np_array])
