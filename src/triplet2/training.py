import torch
import tqdm
from torch import autograd
import numpy as np
import pickle

DIM = 2048
MODE = "complex"

import os
for fname in ["train_acc", "dev_acc"]:

    if os.path.exists(fname):
        os.remove(fname)

def train(model, cca_model, training_generator, dev_generator, loss_fn, cca_loss_fn, optimizer, scheduler, num_epochs):

    SGD = False

    best_acc = 0
    best_loss = 1e7
    training_loss_cca = []
    training_loss_triplet = []
    good, bad = 1e-6, 1e-6

    for epoch in range(num_epochs):


        #loss_fn.k = max(1, 1)

        if epoch  == 500:

            optimizer  = torch.optim.SGD(model.parameters(), weight_decay=0.2 * 1e-7, lr = 0.3 * 1e-2, momentum = 0.9, nesterov = True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4, factor=0.5,
                                                                   verbose=True)
            SGD = True
            print("**********Changing optimizer to Nestrov SGD***********")

        model.zero_grad()

        if epoch >= 0:

            acc, loss, dev_cca_loss, dev_triplet_loss = evaluate(model, cca_model, loss_fn, cca_loss_fn, dev_generator)
            if SGD: scheduler.step(acc)

            with open("TripletModel_last.pickle", "wb") as f:

                pickle.dump(model, f)

            if (acc > best_acc):
                best_acc = acc
                torch.save(model.state_dict(), "TripletModelStateDict.pickle")
                with open("TripletModel.pickle", "wb") as f:
                    pickle.dump(model,f)

            if (loss < best_loss) or 0:
                best_loss = loss

        print()
        print("DEV Acc: {}; DEV loss: {}".format(acc, loss))
        print(" TRAIN acc: {}".format(good / (good + bad)))

        with open("train_acc", "a+") as f:
            f.write("{}\n".format(good / (good + bad)))
        with open("dev_acc", "a+") as f:
            f.write("{}\n".format(acc))

        if cca_loss_fn is not None:
            print("mean DEV cca loss: {}; mean dev triplet loss: {}".format(dev_cca_loss, dev_triplet_loss))
        total_loss = np.array(training_loss_cca) + np.array(training_loss_triplet) if cca_loss_fn is not None else training_loss_triplet

        if training_loss_triplet:
            print("Mean training loss: {}".format(np.mean(total_loss)))
            if cca_loss_fn is not None:
                print ("Mean training cca loss: {}; mean training triplet loss: {}".format(np.mean(training_loss_cca), np.mean(training_loss_triplet)))
        print("\nEpoch {}. Best DEV accuracy so far is {}; best DEV loss so far is: {}".format(epoch, best_acc, best_loss))
        training_loss_cca = []
        training_loss_triplet = []

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator), ascii = True)
        i = 0
        pos_good, pos_bad = 1e-3, 1e-3
        loss_vals = []
        good, bad = 0., 0.

        for (X_padded, Y_padded, X_str, Y_str, lengths, sent_ids) in t:

            i += 1

            if i % 50 == -1:
                print("Evaluting after 50 batches.")
                acc, loss, dev_cca_loss, dev_triplet_loss = evaluate(model, cca_model, loss_fn, cca_loss_fn, dev_generator)

                if (acc > best_acc):
                    best_acc = acc
                    torch.save(model.state_dict(), "TripletModelStateDict.pickle")
                    with open("TripletModel.pickle", "wb") as f:
                        pickle.dump(model, f)


            p1, p2 = model(X_padded, Y_padded, lengths)

            loss, diff, batch_good, batch_bad, norm = loss_fn(p1, p2, X_str, Y_str, sent_ids, 0)

            good += batch_good.detach().cpu().numpy().item()
            bad += batch_bad.detach().cpu().numpy().item()
            training_loss_triplet.append(loss.detach().cpu().numpy().item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            model.zero_grad()

        #print("Position accuracy: {}".format((pos_good / (pos_good + pos_bad))))

def evaluate(model, cca_model, loss_fn, cca_loss_fn, dev_generator):

    print("\nEvaluating...")
    model.eval()
    t = tqdm.tqdm(iter(dev_generator), leave=False, total=len(dev_generator), ascii=True)
    good, bad = 0., 0.
    loss_vals = []
    cca_loss_vals = []
    triplet_loss_vals = []

    norms = []
    diffs = []

    for i, (X_padded, Y_padded, X_str, Y_str, lengths, sent_ids) in enumerate(t):

        with torch.no_grad():

            p1, p2 = model(X_padded, Y_padded, lengths)

            loss, diff, batch_good, batch_bad, norm = loss_fn(p1, p2, X_str, Y_str, sent_ids, i, evaluation = True)

            loss_vals.append(loss.detach().cpu().numpy().item())
            diffs.append(diff.detach().cpu().numpy().item())

            norms.append(norm.detach().cpu().numpy().item())

        good += batch_good
        bad += batch_bad

    good, bad = good.detach().cpu().numpy().item(), bad.detach().cpu().numpy().item()
    acc = good / (good + bad)
    triplet_loss_vals = np.array(loss_vals[:])
    loss_vals = np.array(loss_vals)

    if cca_loss_fn is not None:
        total_loss = loss_vals + cca_loss_vals
    else:
        total_loss = loss_vals

    print("\nMean difference: {}".format(np.mean(diffs)))
    print("Mean norm: {}".format(np.mean(norms)))

    return acc, np.mean(total_loss), np.mean(cca_loss_vals) if cca_loss_vals else [], np.mean(triplet_loss_vals)
