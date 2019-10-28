#!/usr/bin/env bash

# assumes: - wiki.clean.250k and wikipedia.sample.tokenized in data/external

### generate equivalent sentences

#cd data
#mkdir external
#mkdir interim
#cd external

# the following scripts generates both the equivalent sentences file "equivalent_sentences_file.pickle",
# and ELMO/Bert states collected over those sentences, "encoded_sents_file.hdf5".

# with elmo
python src/generate_dataset/main.py --input-wiki data/interim/wikipedia.sample.tokenized \
        --output-data encoded_sents_file.hdf5 \
        --output-sentences data/interim/equivalent_sentences_file.pickle \
        --substitutions-type bert --elmo_folder data/external \
        --cuda-device 0 --dataset-type all --layers 1,2

# with bert
python src/generate_dataset/collect_bert_states.py \
        --input-sentences data/interim/equivalent_sentences/elmo/sents.pickle \
        --output-file data/interim/encoded_bert/sents_bert_base.hdf5 \
        --bert-model bert-base-uncased --layer -1

## collect views for CCA

python src/linear_decomposition/main_views_collector.py \
        --input-path data/interim/encoded_bert/sents_bert_base.hdf5 \
          --num_examples 2000000 --mode simple \
          --output-file data/interim/views/bert_base_layer_2M:-1
# the above script creates a views file in the ./views directory. The file is named according to the arguments (num_examples, etc.)

### run CCA

python src/linear_decomposition/main.py \
        --views-file-path data/interim/views/bert_base_layer_2M:-1 \
        --cca-dim 100 --enforce-symmetry 1 --cca-model numpy \
        --output-file data/processed/models/cca:100_bert_base_layer:-1.pickle
# the above script creates a CCA model in the models directory. The file is named according to the arguments (cca-dim etc.)

### evaluation

# run analysis on the produced embeddings.
python src/analysis/main.py --input-wiki data/external/wiki.clean.250k \
        --encode_sentences true --encoded_data data/interim/eval_encoded_sents_bert_base:-1.pickle \
        --elmo_folder data/external/ --method cosine --cuda-device 0 \
        --num_sents 25000 --num_words 100000 --num_queries 5000 --extractor numpy_cca \
        --extractor_path data/processed/models/cca:100_bert_base_layer:-1

