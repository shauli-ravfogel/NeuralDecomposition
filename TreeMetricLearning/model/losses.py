import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TreeMetricHingeLoss(torch.nn.Module):

        def __init__(self):
        
                super(TreeMetricHingeLoss, self).__init__()
        
        def forward(self, outputs):
        
                """
                Parameters
                ----------
                outputs: a 4X4 tensor, required.
                    we call the indices 0-3 by i,j,k,l. ouputs[i,j] is the L2 distance between nodes i and j.

                Returns
                -------

                loss: torch.FloatTensor, required
                 A scalar loss to be optimised.
                loss = max(0,  (d[i,j] + d[k,l]) - max(d[i,k] + d[j,l], d[i,l] + d[j,k]   ))
                """
        
                inner_max_result = torch.max(outputs[0,2] + outputs[1,3],
                                     outputs[0, 3] + outputs[1, 2])

                loss = torch.max(torch.tensor(0., dtype = torch.float), 
                                (outputs[0,1] + outputs[2,3] - inner_max_result)/inner_max_result)
                return loss

        
        
