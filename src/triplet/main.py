import torch

from torch import optim
from torch.utils import data

import loss
import model
import dataset
import training
#import radam
import math
#import adabound

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

BATCH = 3000
USE_CCA = True
CCA_FINAL_DIM = 2048
TRIPLET_FINAL_DIM = 512
K = 15
MARGIN = 0.05
MODE = "euc"
FINAL = "softmax"

PAIR_REPR = "diff" # diff/abs-diff/product/abs-product/plus

if __name__ == '__main__':

    loss_fn = loss.BatchHardTripletLoss2(alpha = MARGIN, k = K, final = FINAL, mode = MODE)
    cca_loss, cca_network = None, None
    pos_loss = torch.nn.CrossEntropyLoss()
    networks = []

    if USE_CCA:

        cca_network = model.SoftCCANetwork(dim = 2048, final = CCA_FINAL_DIM)
        cca_loss = loss.SoftCCALoss()

    triplet_network = model.Siamese(cca_network, final_dim = TRIPLET_FINAL_DIM, pair_repr = PAIR_REPR).cuda()

    optimizer = optim.Adam(triplet_network.parameters(), weight_decay = 3e-6) # 0 = no weight decay, 1 = full weight decay
    #optimizer = radam.RAdam(network.parameters())
    #optimizer = optim.RMSprop(network.parameters(), weight_decay = 1e-7)
    #optimizer = optim.SGD(network.parameters(), weight_decay=1e-3, lr = 1e-2, momentum = 0.9, nesterov = True)
    #optimizer = adabound.AdaBound(network.parameters(), lr=1e-2, final_lr=0.1)

    #train = dataset.Dataset("sample.15k.pickle")


    train, dev = dataset.Dataset("sample.60k.dist_std=7"), dataset.Dataset("sample.30k.dist_std=7")
    #train, dev = dataset.Dataset("train.dist_std=7"), dataset.Dataset("dev.dist_std=7")

    step_size = 4 * len(train)
    clr = cyclical_lr(step_size)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 4, factor = 0.8, verbose = True)
    training_generator = data.DataLoader(train, batch_size=BATCH, drop_last = False, shuffle=True)
    dev_generator = data.DataLoader(dev, batch_size=BATCH, shuffle=True, drop_last = False)

    training.train(triplet_network, cca_network, training_generator, dev_generator, loss_fn, cca_loss, optimizer, scheduler, num_epochs = 25000)