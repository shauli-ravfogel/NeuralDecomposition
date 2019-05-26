DEFAULT_PARAMS = {"file_name": "resources/wikipedia.sample.tokenized", "pos2words_filename": "resources/pos2words.pickle", "sentences_dict_filename": "resources/sents.pickle", "num_sentences": 10,
"pos_tags_to_replace":  ["NN", "NNS", "NNP", "NNPS", "PRP$", "JJ", "CD"]}

def read_sentences(fname):

	with open(fname, "r") as f:

		lines = f.readlines()

	splitted =  [line.split(" ") for line in lines]
	return splitted
