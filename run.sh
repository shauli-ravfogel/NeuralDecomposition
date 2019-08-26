#!/usr/bin/env bash

# assumes: - wiki.clean.250k and wikipedia.sample.tokenized in data/external

### generate equivalent sentences

cd data
mkdir external
mkdir interim
cd external

python3 src/generate_dataset/main.py --input-wiki data/interim/wikipedia.sample.tokenized --output-data encoded_sents_file.hdf5 --output-sentences data/interim/equivalent_sentences_file.pickle --substitutions-type bert --elmo_folder data/external --cuda-device 0 --dataset-type all --layers 1,2
# the above scripts generates both the equivalent sentences file "equivalent_sentences_file.pickle", and ELMO states collected over those sentences, "encoded_sents_file.hdf5".
# Yanai, please add the BERT version.

## collect views for CCA

cd src/linear_decomposition
python3 collect_views.py --input-path ../../data/interim/encoded_sents_file.hdf5 --num_examples 2000000 --mode simple --exclude_function_words 1
# the above script creates a views file in the ./views directory. The file is named according to the arguments (num_examples, etc.)

### run CCA

python3 main.py --views-file-path ./views/views_file_name --perform-pca 0 --cca-dim 100 --enforce-symmetry 1 --cca-model numpy
# the above script creates a CCA model in the ./models directory. The file is named according to the arguments (cca-dim etc.)

### evaluation

cd ../../analysis
python3 main.py --input-wiki ../data/external/wiki.clean.250k --encode_sentences 0 --encoded_data ../data/interim/encoded_sents.pickle --elmo_folder ../data/external --method cosine --cuda-device 0 --num_sents 25000 --num_words 100000 --num_queries 5000 --extractor numpy_cca --extractor_path ../src/linear_decomposition/models/cca_model_name
# Yanai, the above script run on ELMO states. to Run on BERT states, you need to (1) use extractor trained on BERT states (2) implement embedder.py for BERT (or modify your BERT states collection script so that it will also work on the format of  wiki.clean.250k).