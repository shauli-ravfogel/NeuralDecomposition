import torch
import tqdm
from torch import autograd
import numpy as np

def train(model, training_generator, dev_generator, loss_fn, pos_loss_fn, optimizer, num_epochs=10000):
    best_similarity = 0

    for epoch in range(num_epochs):

        model.zero_grad()
        model.cca.zero_grad()

        if epoch > 0:

            loss = np.mean(loss_vals)
            print("Loss: {}".format(loss))
            similarity = evaluate(model, dev_generator, loss_fn)
            print("Evaluating...(Best similarity so far is {})".format(best_similarity))

            if similarity > best_similarity:
                best_similarity = similarity
                #torch.save(model.state_dict(), "NeuralCCA.pickle")
                torch.save(model, "NeuralCCA.pickle")
                #print(q.cca.mean_x)

        print("Epoch {}".format(epoch))

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator), ascii = True)
        i = 0
        pos_good, pos_bad = 1e-3, 1e-3
        loss_vals = []


        for (view1_vecs, view1_indices), (view2_vecs, view2_indices) in t:

            i += 1

            with autograd.detect_anomaly():
                try:
                    total_corr, (X_proj, Y_proj), pos_pred = model(view1_vecs, view2_vecs)
                except RuntimeError as e:
                    print(e, type(e))
                    continue
                #print(torch.diag(T)[:25])
                #print("---------------------------------")
                loss = loss_fn(X_proj, Y_proj, total_corr)
                loss_vals.append(loss.detach().cpu().numpy())

                """ 
                pos_loss = pos_loss_fn(pos_pred, view1_indices.cuda())
                predicted_indices = torch.argmax(pos_pred, dim = 1).detach().cpu().numpy()
                actual_indices = view1_indices.detach().numpy()
                pos_correct = (predicted_indices == actual_indices)
                pos_good += np.count_nonzero(pos_correct)
                pos_bad += len(pos_correct) - np.count_nonzero(pos_correct)
                loss += 1e-2 * pos_loss
                """

                try:
                    loss.backward()
                except RuntimeError as e:
                        print(e)
                        continue


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
        similarity = 0.
        similarities = []

        with torch.no_grad():

            for (view1_vecs, view1_indices), (view2_vecs, view2_indices) in t:

                X_proj, Y_proj = model.cca(model.layers(view1_vecs), model.layers(view2_vecs), is_training = False)
                cosine_sim = torch.nn.functional.cosine_similarity(X_proj, Y_proj).detach().cpu().numpy().item()
                l2_sim = torch.norm(X_proj - Y_proj, p = 2).detach().cpu().numpy()
                similarities.append((cosine_sim, l2_sim))

            cosine, l2 = list(zip(*similarities))
            cosine_sim = np.mean(cosine)
            l2_dist = np.mean(l2)
            print()
            print("Similarity: ", cosine_sim)
            print("Distance: ", l2_dist)
            return cosine_sim