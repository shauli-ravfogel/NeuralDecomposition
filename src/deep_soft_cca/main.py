import torch

from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt

import loss
import model
import dataset
import training
#import radam

if __name__ == '__main__':

    loss_fn = loss.SoftCCALoss()
    pos_loss = torch.nn.CrossEntropyLoss()
    network = model.Siamese()

    optimizer = optim.Adam(network.parameters(), weight_decay = 5 * 1e-7) # 0 = no weight decay, 1 = full weight decay
    #optimizer = radam.RAdam(network.parameters())
    #optimizer = optim.RMSprop(network.parameters(), weight_decay = 1e-7)
    #optimizer = optim.SGD(network.parameters(), weight_decay=1e-6, lr = 1e-3, momentum = 0.9)
    #train = dataset.Dataset("sample.15k.pickle")
    train = dataset.Dataset("sample.5k")
    dev = dataset.Dataset("sample.5k")

    training_generator = data.DataLoader(train, batch_size=1000, drop_last = True, shuffle=True)
    dev_generator = data.DataLoader(dev, batch_size=1000, shuffle=False, drop_last = True)

    training.train(network, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 25000)