import torch
import tqdm
from torch import autograd
import numpy as np
import pickle

DIM = 2048
MODE = "complex"

def train(model, cca_model, training_generator, dev_generator, loss_fn, cca_loss_fn, optimizer, scheduler, num_epochs):

    SGD = True

    best_acc = 0
    best_loss = 1e7

    for epoch in range(num_epochs):

        #loss_fn.k = max(1, 1)

        if epoch  == 500:

            optimizer  = torch.optim.SGD(model.parameters(), weight_decay=0.2 * 1e-4, lr = 1e-2, momentum = 0.9, nesterov = True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4, factor=0.5,
                                                                   verbose=True)
            SGD = True
            print("**********Changing optimizer to Nestrov SGD***********")

        model.zero_grad()

        if epoch >= 0:

            acc, loss = evaluate(model, cca_model, loss_fn, dev_generator)
            if SGD: scheduler.step(acc)

            if (acc > best_acc):
                best_acc = acc
                torch.save(model.state_dict(), "TripletModelStateDict.pickle")
                with open("TripletModel.pickle", "wb") as f:
                    pickle.dump(model,f)

            if (loss < best_loss) or 0:
                best_loss = loss

        print()
        print("Acc: {}; loss: {}".format(acc, loss))
        print("\nEpoch {}. Best accuracy so far is {}; best loss so far is: {}".format(epoch, best_acc, best_loss))
        #if acc > 0.95: exit()

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator), ascii = True)
        i = 0
        pos_good, pos_bad = 1e-3, 1e-3
        loss_vals = []

        for (w1,w2,w3,w4), sent1, sent2 in t:

            i += 1

            if i % 20 == -1:
                print("Evaluting after 20 batches.")
                acc, loss = evaluate(model, cca_model, loss_fn, dev_generator)

                if (acc > best_acc):
                    best_acc = acc
                    torch.save(model.state_dict(), "TripletModelStateDict.pickle")
                    with open("TripletModel.pickle", "wb") as f:
                        pickle.dump(model, f)


            (w1, w2, w3, w4), (h1, h2, h3, h4), (p1, p2) = model(w1, w3, w2, w4)

            if MODE == "simple":
                loss, diff, batch_good, batch_bad, norm = loss_fn(h1, h2, sent1, sent2, 0) #+ loss_fn(h3, h4, sent1, sent2, 0)
            else:
                loss, diff, batch_good, batch_bad, norm = loss_fn(p1, p2, sent1, sent2, 0)

            if cca_model is not None:

                loss += 4e-1 * 0.5 * (cca_loss_fn(w1, w2) + cca_loss_fn(w3,w4))

                """ 
                pos_loss = pos_loss_fn(pos_pred, view1_indices.cuda())
                predicted_indices = torch.argmax(pos_pred, dim = 1).detach().cpu().numpy()
                actual_indices = view1_indices.detach().numpy()
                pos_correct = (predicted_indices == actual_indices)
                pos_good += np.count_nonzero(pos_correct)
                pos_bad += len(pos_correct) - np.count_nonzero(pos_correct)
                loss += pos_loss
                """

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            model.zero_grad()

        #print("Position accuracy: {}".format((pos_good / (pos_good + pos_bad))))

def evaluate(model, cca_model, loss_fn, dev_generator):

    print("\nEvaluating...")
    model.eval()
    t = tqdm.tqdm(iter(dev_generator), leave=False, total=len(dev_generator), ascii=True)
    good, bad = 0., 0.
    loss_vals = []
    norms = []

    for i, ((w1,w2,w3,w4), sent1, sent2) in enumerate(t):

        with torch.no_grad():
            (w1, w2, w3, w4), (h1, h2, h3, h4), (p1, p2) = model(w1, w3, w2, w4)

            if MODE == "simple":
                loss, diff, batch_good, batch_bad, norm = loss_fn(h1, h2, sent1, sent2, 0) #+ loss_fn(h3, h4, sent1, sent2, 0)
            else:
                loss, diff, batch_good, batch_bad, norm = loss_fn(p1, p2, sent1, sent2, i, evaluation = True)

            loss_vals.append(diff.detach().cpu().numpy().item())
            norms.append(norm.detach().cpu().numpy().item())

        good += batch_good
        bad += batch_bad

    good, bad = good.detach().cpu().numpy().item(), bad.detach().cpu().numpy().item()
    acc = good / (good + bad)
    print("\nMean difference: {}".format(np.mean(loss_vals)))
    print("Mean norm: {}".format(np.mean(norms)))
    return acc, loss