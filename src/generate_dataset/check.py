import pickle
from collections import defaultdict


with open("bert_online_sents_same_pos.pickle", 'rb') as f:
	sents = pickle.load(f)
	
for i in range(min(10000, len(sents))):

	print (" ".join(sents[i][0]))
	
	for j in range(1, len(sents[i])):
	
		print (" ".join(sents[i][j]))
	
	print ("====================================================================")

