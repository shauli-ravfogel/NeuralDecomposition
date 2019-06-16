#!/usr/bin/env bash

# assumes: pos2words.pickle and wikipedia.sample.tokenized files.
# @shauli - where to get these?

cd data
mkdir external
mkdir interim
cd external

wget https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

cd ../../

python src/generate_dataset/main.py --output-data data/interim/pos_data.txt --output-sentences data/interim/pos_sents.pickle --substitutions-type pos
python src/generate_dataset/split_data.py --filename data/processed/emb/emb_data.txt --output-dir data/processed/emb --substitutions-type embeddings

python generate_dataset/split_data.py --filename ../data/processed/pos/pos_data.txt --output-dir ../data/processed/pos/
python generate_dataset/split_data.py --filename ../data/processed/emb/emb_data.txt --output-dir ../data/processed/emb/

cd src

allennlp train framework/experiments/base.jsonnet --include-package framework -s ../allen_logs/pos_base -o '{"trainer.cuda_device": 1}'
allennlp train framework/experiments/siamese_decomposition.jsonnet --include-package framework -s ../allen_logs/pos_decomposition -o '{"trainer.cuda_device": 2}'
