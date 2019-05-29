DEFAULT_PARAMS = {"file_name": "resources/wikipedia.sample.tokenized",
"pos2words_filename": "resources/pos2words.pickle",
"sentences_dict_filename": "resources/sents.pickle",
"dataset_filename": "resources/data.txt",
"elmo_options": "resources/elmo_2x4096_512_2048cnn_2xhighway_options.json",
"elmo_weights": "resources/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
"num_sentences": 20,
"pos_tags_to_replace":  ["NN", "NNS", "NNP", "NNPS", "PRP$", "JJ", "CD"]}

def read_sentences(fname):

	with open(fname, "r") as f:

		lines = f.readlines()

	splitted =  [line.strip().split(" ") for line in lines]
	
	return splitted
	

def to_string(np_array):

	shape = np_array.shape

	return " ".join(["%0.4f" % x for x in np_array])
