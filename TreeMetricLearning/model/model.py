import torch

import numpy as np
from torch import nn
from torch import optim
import tqdm
from torch.utils import data
import matplotlib.pyplot as plt

def distances_between_4_random_vectors(sent_representation, model, p = 2):

        sent_length = len(sent_representation)
        
        perm = torch.randperm(sent_length)
        idx = perm[:4]
        #idx = [0,3,1,6]
        vecs = [sent_representation[idx[0]], sent_representation[idx[1]], sent_representation[idx[2]], sent_representation[idx[3]]]
        
        vecs = model(vecs)
        distances = torch.zeros(4,4)
        
        for i in range(4):
        
                for j in range(4):
                
                        if i == j: continue

                        dis = torch.dist(vecs[i], vecs[j], p)
                        distances[i, j] = 0.1 * dis
        
        if np.random.random() < 0.01:
                print (distances.detach().numpy())
                print("--------------------------------") 
                
        return distances
        
def train(model, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 20):

        for epoch in range(num_epochs):
        
                print("Evaluating...")
                evaluate(model, dev_generator, loss_fn)
                print("Epoch {}".format(epoch))
                
                model.train()

                t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator))
                
                for sent_vecs in t:
                
                        model.zero_grad()
                        distances = distances_between_4_random_vectors(sent_vecs, model)
                        loss = loss_fn(distances)
                        loss.backward()
                        optimizer.step()
                

                

def evaluate(model, eval_generator, loss_fn):

        model.eval()
        good, bad = 0., 0.
        t = tqdm.tqdm(iter(eval_generator), leave = False, total = len(eval_generator))
        average_loss = 0.
        
        with torch.no_grad():

                for sent_vecs in t:

                        distances = distances_between_4_random_vectors(sent_vecs, model)
                        loss = loss_fn(distances).cpu().item()
                        average_loss += loss
                        
                        if abs(loss) < 1e-3:
                                good += 1
                        else:
                                bad += 1
                        
                print(good / (good + bad), average_loss / len(eval_generator))                
                        

class SyntacticTransformation(nn.Module):
        def __init__(self):
                super(SyntacticTransformation, self).__init__()

                layers = []
                layers.append(nn.Linear(1024, 1024, bias = True))
                """
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Linear(512, 512, bias = False))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Linear(512, 1024, bias = False))
                #layers.append(nn.Dropout(0.1))
                """
                
                self.layers = nn.Sequential(*layers)
                
        def forward(self, sent_vecs):

                return [self.layers(v) for v in sent_vecs]
                
                

                        
