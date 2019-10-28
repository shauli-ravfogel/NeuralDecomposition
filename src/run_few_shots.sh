#!/usr/bin/env bash

for train_size in 50 100 200 500 1000
do
  allennlp train few_shots/experiments/constituency_baseline.json --include-package few_shots -s ../allen_logs/base_few_${train_size} -o "{'train_data_path': '/home/nlp/lazary/workspace/thesis/NeuralDecomposition/data/parsing/train_${train_size}'}"
done