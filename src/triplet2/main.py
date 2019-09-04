import torch

from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt

import loss
import model
import dataset
import training


BATCH = 15
USE_CCA = True
CCA_FINAL_DIM = 1024
TRIPLET_FINAL_DIM = 512
K = 70
MARGIN = 0.05
MODE = "euc"
FINAL = "softmax"


if __name__ == '__main__':

    loss_fn = loss.BatchHardTripletLoss2(k=K, mode=MODE, final=FINAL, alpha = MARGIN)
    network = model.Siamese().cuda()

    optimizer = optim.Adam(network.parameters()) # 0 = no weight decay, 1 = full weight decay
    #train = dataset.Dataset("sample.15k.pickle")
    train = dataset.Dataset("sample.pickle")
    dev = dataset.Dataset("sample.pickle")

    training_generator = data.DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=dataset.PadCollate(dim=0))
    dev_generator = data.DataLoader(dev, batch_size=BATCH, shuffle=False, collate_fn=dataset.PadCollate(dim=0))
    training.train(network, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 25000)