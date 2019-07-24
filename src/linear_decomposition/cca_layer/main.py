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
    network = model.ProjectionNetwork()#.cuda()
    optimizer = optim.Adam(network.parameters())
    train = dataset.Dataset("../view1.25.txt", "../view2.25.txt")
    training_generator = data.DataLoader(train, batch_size=25, shuffle=True)
    dev_generator = data.DataLoader(train, batch_size=25, shuffle=False)

    training.train(network, training_generator, dev_generator, loss_fn, optimizer)