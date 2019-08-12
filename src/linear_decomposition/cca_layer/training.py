import torch
import tqdm
from torch import autograd
import numpy as np

def train(model, training_generator, dev_generator, loss_fn, pos_loss_fn, optimizer, num_epochs=10000):
    lowest_loss = 1e9

    for epoch in range(num_epochs):

        model.zero_grad()
        model.cca.zero_grad()

        if epoch > 0:

            loss = np.mean(loss_vals)
            print("Loss: {}".format(loss))
            #loss = evaluate(model, dev_generator, loss_fn)
            #print("Evaluating...(Lowest dev set loss so far is {})".format(lowest_loss))

            if loss < lowest_loss:
                lowest_loss = loss
                #torch.save(model.state_dict(), "NeuralCCA.pickle")
                torch.save(model, "NeuralCCA.pickle")
                #print(q.cca.mean_x)

        print("Epoch {}".format(epoch))

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator))
        t = iter(training_generator)
        i = 0
        pos_good, pos_bad = 1e-3, 1e-3
        loss_vals = []


        for (view1_vecs, view1_indices), (view2_vecs, view2_indices) in t:

            i += 1

            with autograd.detect_anomaly():
                try:
                    total_corr, (X_proj, Y_proj), pos_pred = model(view1_vecs, view2_vecs)
                except RuntimeError:
                    print("Error.")
                    continue
                #print(torch.diag(T)[:25])
                #print("---------------------------------")
                loss = loss_fn(X_proj, Y_proj, total_corr)
                loss_vals.append(loss.detach().cpu().numpy())
                pos_loss = pos_loss_fn(pos_pred, view1_indices.cuda())
                predicted_indices = torch.argmax(pos_pred, dim = 1).detach().cpu().numpy()
                actual_indices = view1_indices.detach().numpy()
                pos_correct = (predicted_indices == actual_indices)
                pos_good += np.count_nonzero(pos_correct)
                pos_bad += len(pos_correct) - np.count_nonzero(pos_correct)
                loss += pos_loss

                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            model.zero_grad()
            model.cca.zero_grad()

        print("Position accuracy: {}".format((pos_good / (pos_good + pos_bad))))

def evaluate(model, eval_generator, loss_fn):

        model.eval()
        good, bad = 0., 0.
        #t = tqdm.tqdm(iter(eval_generator), leave=False, total=len(eval_generator))
        t = iter(eval_generator)
        average_loss = 0.

        with torch.no_grad():

            for (view1_vecs, view1_indices), (view2_vecs, view2_indices) in t:

                total_corr, (X_proj, Y_proj), pos_pred = model(view1_vecs, view2_vecs)
                loss = loss_fn(X_proj, Y_proj, total_corr)
                average_loss += loss


            average_loss /= len(eval_generator)

            print("LOSS: ", average_loss)
            return average_loss