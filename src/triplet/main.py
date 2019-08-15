import torch

from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt

import loss
import model
import dataset
import training


if __name__ == '__main__':

    loss_fn = loss.TripletLoss()
    pos_loss = torch.nn.CrossEntropyLoss()
    network = model.Siamese().cuda()

    optimizer = optim.Adam(network.parameters()) # 0 = no weight decay, 1 = full weight decay
    train = dataset.Dataset("data.pickle")
    training_generator = data.DataLoader(train, batch_size=100, shuffle=True)
    dev_generator = data.DataLoader(train, batch_size=50, shuffle=False)

    training.train(network, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 100)