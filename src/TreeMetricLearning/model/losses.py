import torch
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


if __name__ == '__main__':

    loss_func = TreeMetricHingeLoss()

    for i in range(1000):
        distances = np.random.rand(4, 4) - 0.5
        i, j, k, l = 0, 1, 2, 3
        d_ij = distances[i, j]
        d_kl = distances[k, l]
        d_ik = distances[i, k]
        d_jl = distances[j, l]
        d_il = distances[i, l]
        d_jk = distances[j, k]

        expected_loss = np.maximum(0, d_ij + d_kl - np.maximum(d_ik + d_jl, d_il + d_jk))
        outputs = torch.from_numpy(distances).type(torch.float)
        loss = loss_func(outputs).item()

        assert np.isclose(loss, expected_loss, atol=1e-4)

    print("Test ended successfully.")