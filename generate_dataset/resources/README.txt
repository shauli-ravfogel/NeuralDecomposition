data.txt format:

indices tab sent1 tab sent2 tab vecs1 tab vecs2 tab label

---------------------------------------------------------------------------------------------------------------------------
indices: the indices in the sentence chosen for this example, space separated. e.g., if indices = 1 3
	then we collect vectors over the first and the third words in the sentence.
	The default is just a single index (e.g. 6).
sent1: the first sentence, space separated.
sent2: the second sentence, space separated.
vecs1: the vectors over the chosen indices, in the first sentence. individual number are space separated,
	and the symbol * separates between different vectors (example: 1.3 0.1 5 ...*3.3 0.4 5...). In the default 		setting, indices is just a single number (e.g. 5),
	and vecs1 is a single vector (e.g. ELMO vector over the 5th word).
vec2: as above, for the vectors of the second sentence.
label: 0 or 1 (syntactically equivalent or not)
