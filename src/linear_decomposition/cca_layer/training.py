import torch
import tqdm
from torch import autograd

def train(model, training_generator, dev_generator, loss_fn, optimizer, num_epochs=2000):
    lowest_loss = 1e9

    for epoch in range(num_epochs):

        model.zero_grad()
        model.cca.zero_grad()

        if epoch > 0:
            loss = evaluate(model, dev_generator, loss_fn)
            print("Evaluating...(Lowest dev set loss so far is {})".format(lowest_loss))

            if loss < lowest_loss:
                lowest_loss = loss
                torch.save(model.state_dict(), "NeuralCCA.pickle")

        print("Epoch {}".format(epoch))

        model.train()

        t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator))
        t = iter(training_generator)
        i = 0

        for view1, view2 in t:

            i += 1

            with autograd.detect_anomaly():

                T, (X_proj, Y_proj) = model(view1, view2)
                loss = loss_fn(X_proj, Y_proj, T)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            model.zero_grad()
            model.cca.zero_grad()

def evaluate(model, eval_generator, loss_fn):

        model.eval()
        good, bad = 0., 0.
        #t = tqdm.tqdm(iter(eval_generator), leave=False, total=len(eval_generator))
        t = iter(eval_generator)
        average_loss = 0.

        with torch.no_grad():

            for view1, view2 in t:

                T, (X_proj, Y_proj) = model(view1, view2)
                loss = loss_fn(X_proj, Y_proj, T)
                average_loss += loss


            average_loss /= len(eval_generator)

            print("LOSS: ", average_loss)
            return average_loss