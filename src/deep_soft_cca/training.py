import torch
import tqdm
from torch import autograd
import numpy as np
import pickle

def train(model, training_generator, dev_generator, loss_fn, optimizer, scheduler, num_epochs):

    best_loss = 1e6

    for epoch in range(num_epochs):

        model.zero_grad()

        if epoch >= 0:

            loss = evaluate(model, loss_fn, dev_generator)
            #scheduler.step(loss)

            if (loss < best_loss) or False:
                best_loss = loss
                torch.save(model.state_dict(), "NeuralCCAStateDict.pickle")
                torch.save(model, "NeuralCCA.pickle")
        print()
        print("Loss: {}".format(loss))
        print("\nEpoch {}. Best loss so far is {}".format(epoch, best_loss))

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator), ascii = True)
        i = 0

        for (w1,w2,w3,w4) in t:

            view1, view2 = (w1, w3) if np.random.random() < 0.5 else (w2,w4)
            X,Y =  model(view1,view2)
            loss = loss_fn(X,Y)
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

    for (w1,w2,w3,w4) in t:

        with torch.no_grad():

            view1, view2 = (w1, w3) if np.random.random() < 0.5 else (w2,w4)
            X,Y =  model(view1,view2)
            loss = loss_fn(X,Y)
            loss_vals.append(loss.detach().cpu().numpy().item())

    return np.mean(loss_vals)