import torch
import tqdm
from torch import autograd
import numpy as np
import pickle

DIM = 2048
MODE = "complex"

def train(model, training_generator, dev_generator, loss_fn, optimizer, scheduler, num_epochs):

    best_acc = 0
    for epoch in range(num_epochs):

        loss_fn.k = max(1, 1)

        #if epoch % 15 == 0 and epoch > 0:

        #    optimizer = torch.optim.Adam(model.parameters(), weight_decay=5 * 1e-4)

        model.zero_grad()

        if epoch >= 0:

            acc, loss = evaluate(model, loss_fn, dev_generator)
            #scheduler.step(acc)
            if (acc > best_acc) or 0:
                best_acc = acc
                torch.save(model.state_dict(), "TripletModelStateDict.pickle")
                with open("TripletModel.pickle", "wb") as f:
                    pickle.dump(model,f)
        print()
        print("Acc: {}".format(acc))
        print("\nEpoch {}. Best accuracy so far is {}".format(epoch, best_acc))
        #if acc > 0.95: exit()

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator), ascii = True)
        i = 0
        pos_good, pos_bad = 1e-3, 1e-3
        loss_vals = []

        for (w1,w2,w3,w4,w5,w6, w7, w8, w9, w10) in t:
            w1, w3, w5 = w1[:, :DIM], w3[:, :DIM], w5[:, :DIM]

            i += 1

            with autograd.detect_anomaly():
                try:

                    #p1 = model(w1,w4)
                    #p2 = model(w3,w2)
                    #p3 = model(w5, w8)

                    if np.random.random() < 0.0:
                        p1 = model(w1,w2)
                        p2 = model(w3,w4)
                    else:
                        p1 = model(w1, w4)
                        p2 = model(w3, w2)


                    p3 = model(w5, w2)

                except RuntimeError as e:
                    print(e, type(e))
                    exit()

                if MODE == "simple":
                    loss, diff, batch_good, batch_bad, norm = loss_fn(model.process(w1),model.process(w3),
                                                                     model.process(w5))
                else:
                    loss, diff, batch_good, batch_bad, norm = loss_fn(p1, p2, p3)


                #loss, good, bad = loss_fn(model.final_net(w1), model.final_net(w3), model.final_net(w5))
                #dis_ll = (1 - torch.nn.functional.cosine_similarity(model.layers(w1), model.layers(w3))).sum() #torch.norm(model(w1,w3), dim = 1, p = 2).sum()
                #dis_mm  = (1 - torch.nn.functional.cosine_similarity(model.layers(w2), model.layers(w4))).sum() #torch.norm(model(w2,w4), dim = 1, p = 2).sum()
                #dis_ll = (torch.norm(model.final_net(w1) - model.final_net(w3), dim = 1, p = 2)**2).sum()
                #dis_mm = (torch.norm(model.final_net(w2) - model.final_net(w4), dim = 1, p = 2)**2).sum()
                #loss = loss + 1e-1 * (dis_ll + dis_mm)
                #loss, batch_good, batch_bad = loss_fn(model.final_net(w1), model.final_net(w3), model.final_net(w5))

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

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)

            optimizer.step()
            model.zero_grad()

        #print("Position accuracy: {}".format((pos_good / (pos_good + pos_bad))))

def evaluate(model, loss_fn, dev_generator):

    print("\nEvaluating...")
    model.eval()
    t = tqdm.tqdm(iter(dev_generator), leave=False, total=len(dev_generator), ascii=True)
    good, bad = 0., 0.
    loss_vals = []
    norms = []

    for (w1,w2,w3,w4,w5,w6, w7, w8, w9, w10) in t:

        w1, w3, w5 = w1[:, :DIM], w3[:, :DIM], w5[:, :DIM]

        with torch.no_grad():
            #p1 = model(w1, w4)
            #p2 = model(w3, w2)
            #p3 = model(w5, w8)

            if np.random.random() < 0.0:
                p1 = model(w1, w2)
                p2 = model(w3, w4)
            else:
                p1 = model(w1, w4)
                p2 = model(w3, w2)

            p3 = model(w5, w2)

            if MODE == "simple":
                loss, diff, batch_good, batch_bad, norm = loss_fn(model.process(w1), model.process(w3),
                                                                  model.process(w5))
            else:
                loss, diff, batch_good, batch_bad, norm = loss_fn(p1, p2, p3)

            loss_vals.append(diff.detach().cpu().numpy().item())
            norms.append(norm.detach().cpu().numpy().item())

        good += batch_good
        bad += batch_bad

    good, bad = good.detach().cpu().numpy().item(), bad.detach().cpu().numpy().item()
    acc = good / (good + bad)
    print("\nMean difference: {}".format(np.mean(loss_vals)))
    print("Mean norm: {}".format(np.mean(norms)))
    return acc, loss