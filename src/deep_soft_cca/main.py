import torch

from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt

import loss
import model
import dataset
import training
import radam

if __name__ == '__main__':

    loss_fn = loss.SoftCCALoss()
    pos_loss = torch.nn.CrossEntropyLoss()
    network = model.Siamese().cuda()

    #optimizer = optim.Adam(network.parameters()) # 0 = no weight decay, 1 = full weight decay
    #optimizer = optim.Adagrad(network.parameters(), lr = 1e-2)
    #optimizer = radam.RAdam(network.parameters())
    #optimizer = optim.RMSprop(network.parameters())
    optimizer = optim.SGD(network.parameters(), lr = 0.3 * 1e-1, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor = 0.5, verbose = True)
    #train = dataset.Dataset("sample.15k.pickle")
    train, dev = dataset.Dataset("sample.60k.no_func"), dataset.Dataset("sample.30k.no_func")
    #train, dev = dataset.Dataset("train1.pickle"), dataset.Dataset("dev1.pickle")
    training_generator = data.DataLoader(train, batch_size=2000, drop_last = False, shuffle=True)
    dev_generator = data.DataLoader(dev, batch_size=2000, shuffle=False, drop_last = False)

    training.train(network, training_generator, dev_generator, loss_fn, optimizer, scheduler, num_epochs = 25000)