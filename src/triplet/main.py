import torch

from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt

import loss
import model
import dataset
import training
import radam
import math
import adabound

def cyclical_lr(stepsize, min_lr=3 * 1e-4, max_lr=1e-1):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

if __name__ == '__main__':

    loss_fn = loss.BatchHardTripletLoss()
    pos_loss = torch.nn.CrossEntropyLoss()
    network = model.Siamese()

    #optimizer = optim.Adam(network.parameters(), weight_decay = 5 * 1e-7) # 0 = no weight decay, 1 = full weight decay
    #optimizer = radam.RAdam(network.parameters())
    #optimizer = optim.RMSprop(network.parameters(), weight_decay = 1e-7)
    optimizer = optim.SGD(network.parameters(), weight_decay=1e-5, lr = 5 * 1e-1, momentum = 0.9, nesterov = True)
    optimizer = adabound.AdaBound(network.parameters(), lr=1e-3, final_lr=0.1)

    #train = dataset.Dataset("sample.15k.pickle")


    train = dataset.Dataset("sample3.60k")
    dev = dataset.Dataset("sample3.5k")

    step_size = 4 * len(train)
    clr = cyclical_lr(step_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 3, factor = 0.5)
    training_generator = data.DataLoader(train, batch_size=64, drop_last = True, shuffle=True)
    dev_generator = data.DataLoader(dev, batch_size=64, shuffle=False, drop_last = True)

    training.train(network, training_generator, dev_generator, loss_fn, optimizer, scheduler, num_epochs = 25000)