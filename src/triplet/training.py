import torch
import tqdm
from torch import autograd
import numpy as np

def train(model, training_generator, dev_generator, loss_fn, optimizer, num_epochs):

    best_acc = 0

    for epoch in range(num_epochs):

        model.zero_grad()

        if epoch >= 0:

            acc = evaluate(model, loss_fn, dev_generator)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "TripletModelStateDict.pickle")
                torch.save(model, "TripletModel.pickle")

        print("\nEpoch {}. Best accuracy so far is {}".format(epoch, best_acc))

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator), ascii = True)
        i = 0
        pos_good, pos_bad = 1e-3, 1e-3
        loss_vals = []

        for (w1,w2,w3,w4,w5,w6) in t:

            i += 1

            with autograd.detect_anomaly():
                try:
                    p1 = model(w1,w2)
                    p2 = model(w3,w4)
                    p3 = model(w5, w6)
                except RuntimeError as e:
                    print(e, type(e))
                    exit()

                loss, good, bad = loss_fn(p1, p2, p3)
                #loss_vals.append(loss.detach().cpu().numpy())
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            model.zero_grad()

        #print("Position accuracy: {}".format((pos_good / (pos_good + pos_bad))))

def evaluate(model, loss_fn, dev_generator):

    print("\nEvaluating...")
    model.eval()
    t = tqdm.tqdm(iter(dev_generator), leave=False, total=len(dev_generator), ascii=True)
    good, bad = 0., 0.

    for (w1, w2, w3, w4, w5, w6) in t:

        with torch.no_grad():

            p1 = model(w1, w2)
            p2 = model(w3, w4)
            p3 = model(w5, w6)
            loss, batch_good, batch_bad = loss_fn(p1, p2, p3)

        good += batch_good
        bad += batch_bad

    good, bad = good.detach().cpu().numpy().item(), bad.detach().cpu().numpy().item()
    acc = good / (good + bad)

    return acc