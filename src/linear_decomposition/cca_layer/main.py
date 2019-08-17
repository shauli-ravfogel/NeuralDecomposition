import torch

import numpy as np
from torch import nn
from torch import optim
import tqdm
from torch.utils import data
import matplotlib.pyplot as plt

import loss
import model
import dataset
import training
if __name__ == '__main__':

    loss_fn = loss.SimilarityLoss()
    pos_loss = torch.nn.CrossEntropyLoss()
    network = model.ProjectionNetwork().cuda()
    network.cca.cuda()

    optimizer = optim.Adam(network.parameters(), weight_decay = 0.5 * 1e-4) # 0 = no weight decay, 1 = full weight decay
    train = dataset.Dataset("../sample.45k.pickle")
    dev = dataset.Dataset("../sample.5k.pickle")
    training_generator = data.DataLoader(train, batch_size=5000, shuffle=True)
    dev_generator = data.DataLoader(dev, batch_size=1, shuffle=False)

    training.train(network, training_generator, dev_generator, loss_fn, pos_loss, optimizer)