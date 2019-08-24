import torch

from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt

import loss
import model
import dataset
import training


if __name__ == '__main__':

    loss_fn = loss.BatchHardTripletLoss()
    pos_loss = torch.nn.CrossEntropyLoss()
    network = model.Siamese().cuda()

    optimizer = optim.Adam(network.parameters()) # 0 = no weight decay, 1 = full weight decay
    #optimizer = optim.RMSprop(network.parameters(), weight_decay = 1e-4)
    #optimizer = optim.SGD(network.parameters(), weight_decay=1e-4, lr = 1e-2, momentum = 0.9)
    #train = dataset.Dataset("sample.15k.pickle")
    train = dataset.Dataset("sample3.25k")
    dev = dataset.Dataset("sample3.5k")

    training_generator = data.DataLoader(train, batch_size=128, shuffle=True)
    dev_generator = data.DataLoader(dev, batch_size=128, shuffle=False)

    training.train(network, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 25000)