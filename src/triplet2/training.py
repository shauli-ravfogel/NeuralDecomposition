import torch
import tqdm
from torch import autograd
import numpy as np
import pickle

def train(model, training_generator, dev_generator, loss_fn, optimizer, num_epochs):

    best_acc = 0

    for epoch in range(num_epochs):

        model.zero_grad()

        if epoch > 0:

            acc = evaluate(model, loss_fn, dev_generator)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "TripletModelStateDict2.pickle")
                with open("TripletModel2.pickle", "wb") as f:
                    pickle.dump(model,f)

        print("\nEpoch {}. Best accuracy so far is {}".format(epoch, best_acc))
        #if acc > 0.95: exit()

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator), ascii = True)
        i = 0
        pos_good, pos_bad = 1e-3, 1e-3
        loss_vals = []

        for sent1_vecs in t:

            print(sent1_vecs.shape)
            exit()
            model(sent1_vecs)

            l = min(p1.shape[1], p3.shape[1])
            p1,p2,p3 = p1[:, :l], p2[:, :l], p3[:, :l]
            loss, _, _ = loss_fn(p1, p2, p3, sent1, sent2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.)
            optimizer.step()
            model.zero_grad()

        #print("Position accuracy: {}".format((pos_good / (pos_good + pos_bad))))

def evaluate(model, loss_fn, dev_generator):

    print("\nEvaluating...")
    model.eval()
    t = tqdm.tqdm(iter(dev_generator), leave=False, total=len(dev_generator), ascii=True)
    good, bad = 0., 0.
    total_loss = []

    for sent1_vecs, sent2_vecs in t:

        i, j = np.random.choice(range(15), replace=False, size=2)
        pos_1, pos_2, neg = sent1_vecs[:,i, ...], sent1_vecs[:, j, ...], sent2_vecs[:, i, ...]

        with torch.no_grad():

            (p1, sent1), (p2, sent2), (p3, sent3) = model(pos_1), model(pos_2), model(neg)
            l = min(p1.shape[1], p3.shape[1])
            p1,p2,p3 = p1[:, :l], p2[:, :l], p3[:, :l]
            loss, batch_good, batch_bad = loss_fn(p1, p2, p3, sent1, sent2)
            total_loss.append(loss.detach().cpu().numpy().item())

        good += batch_good
        bad += batch_bad

    good, bad = good.detach().cpu().numpy().item(), bad.detach().cpu().numpy().item()
    acc = good / (good + bad)
    print("\nLoss: {}".format(np.mean(total_loss)))
    return acc