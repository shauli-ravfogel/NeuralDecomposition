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


BATCH = 200
USE_CCA = False
CCA_FINAL_DIM = 1024
TRIPLET_FINAL_DIM = 128
K = 40
MARGIN = 0.1
MODE = "cosine"
FINAL = "softmax"
FILTER_FUNC_WORDS = False

if __name__ == '__main__':

    loss_fn = loss.BatchHardTripletLoss2(alpha = MARGIN, k = K, final = FINAL, mode = MODE)
    cca_loss, cca_network = None, None
    pos_loss = torch.nn.CrossEntropyLoss()
    networks = []

    if USE_CCA:

        cca_network = model.SoftCCANetwork(dim = 2048, final = CCA_FINAL_DIM)
        cca_loss = loss.SoftCCALoss()

    triplet_network = model.ELMOEncoder(layer_sizes = [1024, 512])

    optimizer = optim.Adam(triplet_network.parameters(), weight_decay = 1e-6) # 0 = no weight decay, 1 = full weight decay
    #optimizer = radam.RAdam(network.parameters())
    #optimizer = optim.RMSprop(network.parameters(), weight_decay = 1e-7)
    #optimizer = optim.SGD(network.parameters(), weight_decay=1e-3, lr = 1e-2, momentum = 0.9, nesterov = True)
    #optimizer = adabound.AdaBound(network.parameters(), lr=1e-2, final_lr=0.1)

    #train = dataset.Dataset("sample.15k.pickle")


    train, dev = dataset.Dataset("bert_online_sents_same_pos5.pickle", filter_func = FILTER_FUNC_WORDS), dataset.Dataset("bert_online_sents_same_pos5.pickle", filter_func = FILTER_FUNC_WORDS)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 4, factor = 0.8, verbose = True)
    training_generator = data.DataLoader(train, batch_size=BATCH, drop_last = False, shuffle=True, collate_fn=dataset.PadCollate())
    dev_generator = data.DataLoader(dev, batch_size=BATCH, drop_last = False, shuffle=True, collate_fn=dataset.PadCollate())
    training.train(triplet_network, cca_network, training_generator, dev_generator, loss_fn, cca_loss, optimizer, scheduler, num_epochs = 25000)