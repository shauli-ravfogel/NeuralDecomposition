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



if __name__ == '__main__':


        loss_fn = losses.TreeMetricHingeLoss()
        network = model.SyntacticTransformation().cuda()
        optimizer = optim.Adam(network.parameters())
        #optimizer = optim.SGD(network.parameters(), lr = 0.001, momentum=0.9)
        train = dataset.Dataset("../data/processed/train")
        dev = dataset.Dataset("../data/processed/dev")
        training_generator = data.DataLoader(train, batch_size = 1, shuffle = True)
        dev_generator = data.DataLoader(dev, batch_size = 1, shuffle = False)
        
        model.train(network, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 25000)
