import torch

import numpy as np
from torch import nn
from torch import optim
import tqdm
from torch.utils import data
import matplotlib.pyplot as plt

import losses
import model
import dataset
import torch.backends.cudnn

torch.backends.cudnn.benchmark=True


if __name__ == '__main__':

        loss_fn = losses.ELMORecoveryLoss().cuda()
        #loss_fn = losses.CorrelationMSTMetricLoss()
        elmo_options = "../data/external/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        elmo_weights = "../data/external/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                
        network = model.SyntacticTransformation()
        network.cuda()
        
        optimizer = optim.Adam(network.parameters())
        #optimizer = optim.SGD(network.parameters(), momentum=0.5, lr = 0.01, weight_decay = 0.005)
        train = dataset.Dataset("../data/internal/bert_online_sents_maintain_pos.pickle", elmo_options = elmo_options, elmo_weights = elmo_weights)
        dev = dataset.Dataset("../data/internal/bert_online_sents_maintain_pos.pickle", elmo_options = elmo_options, elmo_weights = elmo_weights)
        training_generator = data.DataLoader(train, batch_size = 1, shuffle = True)
        dev_generator = data.DataLoader(dev, batch_size = 1, shuffle = False)

        model.train(network, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 25000)
