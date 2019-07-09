import torch

import numpy as np
from torch import nn
from torch import optim
import tqdm
from torch.utils import data
import matplotlib.pyplot as plt
from torch.nn.modules.distance import PairwiseDistance
        
def train(model, training_generator, dev_generator, loss_fn, optimizer, num_epochs = 20, update_freq = 1):

        best_acc = 1e9
        
        for epoch in range(num_epochs):
        
                model.zero_grad()
                print("Evaluating...(Best dev set accuracy so far is {})".format(best_acc))
                if epoch > 0: 
                        acc = evaluate(model, dev_generator, loss_fn)
                        if acc < best_acc:
                
                                best_acc = acc
                                torch.save(model.state_dict(), "model.pickle")

                print("Epoch {}".format(epoch))
                
                model.train()

                t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator))
                
                i = 0
                loss = torch.zeros([1, 1])
                
                for (embds, layer1, layer2) in t:
                
                        i += 1
                                            
                        #distances = distances_between_4_random_vectors(sent_vecs, model)
                        #loss = loss_fn(distances)
                        loss = loss_fn(embds, layer1, layer2, model)
                        
                        if i % update_freq == 0:
                                 
                                 loss /= update_freq
                                 loss.backward()
                                 torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
                                 optimizer.step()
                                 model.zero_grad()
                                 loss = torch.zeros([1, 1])                

                

def evaluate(model, eval_generator, loss_fn):

        model.eval()
        good, bad = 0., 0.
        t = tqdm.tqdm(iter(eval_generator), leave = False, total = len(eval_generator))
        average_loss = 0.
        
        with torch.no_grad():

                for (embds, layer1, layer2) in t:

                        #distances = distances_between_4_random_vectors(sent_vecs, model)
                        #loss = loss_fn(distances).cpu().item()
                        loss = loss_fn(embds, layer1, layer2, model).item()
                        average_loss += loss
                        """
                        if abs(loss) < 1e-3:
                                good += 1
                        else:
                                bad += 1
                        """
                
                average_loss /= len(eval_generator)
                        
                #acc, avg_loss =   good / (good + bad), average_loss / len(eval_generator)
                #print(good / (good + bad), average_loss / len(eval_generator))
                print(average_loss)
                return average_loss              
                        

class SyntacticTransformation(nn.Module):
        def __init__(self):
                super(SyntacticTransformation, self).__init__()

                layers = []
                layers.append(nn.Linear(1024, 512, bias = False)) 
                layers.append(nn.ReLU())
                layers.append(nn.Linear(512, 1024, bias = False))             
                self.initial_transform_model = nn.Sequential(*layers)



                layers = []
                layers.append(nn.Linear(1024, 512, bias = False)) 
                layers.append(nn.ReLU())
                layers.append(nn.Linear(512, 1024, bias = False))
              
                self.values_weights = nn.Sequential(*layers)
                
                
                
                layers = []
                layers.append(nn.Linear(1024, 512, bias = False)) 
                layers.append(nn.ReLU())
                layers.append(nn.Linear(512, 1024, bias = False))
              
                self.keys_weights = nn.Sequential(*layers)
                
                
                
                layers = []
                layers.append(nn.Linear(1024, 512, bias = False)) 
                layers.append(nn.ReLU())
                layers.append(nn.Linear(512, 1024, bias = False))
              
                self.semantics_weights = nn.Sequential(*layers)                
                
                                               
        def initial_transform(self, sent_vecs):

                return self.initial_transform_model(sent_vecs)

        def values(self, sent_vecs):

                return self.values_weights(sent_vecs)
                
        def keys(self, sent_vecs):

                return self.keys_weights(sent_vecs)
                
        def semantic_transformation(self, sent_vecs):
        
                return self.semantics_weights(sent_vecs)
                                                
        def get_distances_matrix(self, sent_vecs):
                       
                sent_vecs_transformed = self.forward(sent_vecs)
                distances_after = PairwiseDistance(p = 2).forward(sent_vecs, sent_vecs)
                return distances 
                

                        
