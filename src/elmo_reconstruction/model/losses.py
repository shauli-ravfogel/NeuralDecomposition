import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.distance import PairwiseDistance
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path, dijkstra
import model
import scipy.spatial
from torch import optim
np.set_printoptions(precision=3)


class ELMORecoveryLoss(torch.nn.Module):

        def __init__(self):
        
                super(ELMORecoveryLoss, self).__init__()
                self.softmax = nn.LogSoftmax(dim = 0)
        
        def forward(self, embds, layer1, layer2, model):
        
                embds, layer1, layer2 = embds[0].cuda(), layer1[0].cuda(), layer2[0].cuda()
                #print(embds.shape, layer1.shape, layer2.shape)
                sents_length = embds.shape[0]
                i,j = np.random.choice(range(sents_length), size = 2, replace = False)
                embds_i = embds[i]
                elmo_i = layer1[i] + layer2[i]
                elmo_j = layer1[j] + layer2[j]
                
                elmo_j_transformed = model.initial_transform(elmo_i)
                elmo_j_keys = model.keys(elmo_j_transformed)
                elmo_j_values = model.values(elmo_j_transformed)
                
                elmo_j_weights = torch.nn.functional.softmax(torch.mm(elmo_j_keys, torch.t(elmo_j_values)), dim = 1)
                #print(elmo_j_weights[0], torch.sum(elmo_j_weights[0]), torch.sum(elmo_j_weights[:,0]), elmo_j_weights.shape)
                #exit()
                semantics = model.semantic_transformation(torch.t(torch.mm(torch.t(embds_i),elmo_j_weights)))
                
                #print(elmo_j_transformed.shape, elmo_j_keys.shape,  elmo_j_values.shape, elmo_j_weights.shape, semantics.shape)
                loss = torch.sum((elmo_i - (semantics + elmo_j_transformed))**2)
                
                return loss







                        
if __name__ == '__main__':
        """
        loss_func = TreeMetricHingeLoss()
        
        for i in range(1000):
        
                distances = np.random.rand(4,4) - 0.5
                i,j,k,l = 0,1,2,3
                d_ij = distances[i,j]
                d_kl = distances[k,l]
                d_ik = distances[i,k]
                d_jl = distances[j,l]
                d_il = distances[i,l]
                d_jk = distances[j,k]

                expected_loss = np.maximum(0, d_ij + d_kl - np.maximum(d_ik + d_jl, d_il + d_jk))
                outputs = torch.from_numpy(distances).type(torch.float)
                loss = loss_func(outputs).item()

                assert np.isclose(loss, expected_loss, atol = 1e-4)
                
        print("Test ended successfully.")
        """  
           
        loss_func = MSTMetricLoss()
        n = 5
        dim = 1024

        np.random.seed(0)
        vecs_np = [(np.random.rand(dim)) for i in range(n)]
        vecs_torch = [torch.from_numpy(v).type(torch.float) for v in vecs_np]
        distances_np = scipy.spatial.distance_matrix(vecs_np, vecs_np)
        tree = minimum_spanning_tree(distances_np)
        tree_paths = dijkstra(tree, directed = False)
        tree_paths2 = shortest_path(tree)

        network = model.SyntacticTransformation()
        network.zero_grad()
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
        loss_func(vecs_torch, network)
        
        
