
import numpy as np
from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
import scipy
np.set_printoptions(precision=3)
import random
import pickle
import tqdm
from sklearn.cross_decomposition import CCA


class CCADecomposition(object):

        def __init__(self, data_filename):
        
                self.data_filename = data_filename
                self.data, (self.view1, self.view2) = self.collect_data()
                self.perform_cca()

                                        
        def _print_after_projection(self, pca_before, pca_after):
        
                num_groups = 1000
                avg_before = np.zeros(1024)
                avg_after = np.zeros(1024)
                
                for group in self.groups[:num_groups]: # each group is the ith WORD in one group of equivalent sentences

                        # components = self.components (num_componenets, 1024)

                        projected = pca_after.transform(group) # (num_components, num_equivalent_sents)
                        semantic_proj = pca_after.components_        
                        semantic_components = semantic_proj.T.dot(semantic_proj.dot(group.T))
 
                        syntactic_proj = scipy.linalg.orth(semantic_proj)

                        syntactic_componet = group.T - semantic_components # (num_equivalent_sents, 1024)

                        group = group.T # (1024, num_equivalent_sents)
                        
                        avg_std_before = np.std(group, axis = 1) # (1024, )
                        avg_std_after = np.std(syntactic_componet, axis = 1) # (1024, )
                        
                        avg_before += avg_std_before
                        avg_after += avg_std_after

                avg_before /= num_groups
                avg_after /= num_groups
                

                        
                print(avg_before[:50])
                print(avg_after[:50])
                        
        def _plot(self, var_before, var_after):
        
                print(len(var_before), len(var_after))
                fig = plt.figure()
                plt.plot(range(len(var_before)), var_before, color = "blue", label = "Before substraction of mean")
                plt.plot(range(len(var_after)), var_after, color = "red", label = "After substraction of mean")
                plt.xlabel("Index of principal components")
                plt.ylabel("explaned variance")
                plt.title("explained vairance vs. number of principal components")
                plt.legend()
                plt.show()

        def collect_data(self, n = 500):
        
                print("Collecting data from {} sentences...".format(n))
                
                A, B = [], [] # view matrices
                
                with open(self.data_filename, "r") as f:
                
                        vecs = []
                        
                        for i in tqdm.tqdm(range(n)):
                        
                               line = f.readline()
                               data, sents = self.read_one_set(line)
                               
                               for sent_index in range(data.shape[1]):
                                       
                                       group = data[:, sent_index, :] # (num_equivalent_sents, 1024)
                                       view1, view2 = group[::2, :], group[1::2, :]
                                       q = min(view1.shape[0], view2.shape[0])
                                       view1, view2 = view1[:q, :], view2[:q, :]
                                       A.extend([view1[i] for i in range(view1.shape[0])])
                                       B.extend([view2[i] for i in range(view1.shape[0])])     
                                       vecs.extend([group[i].copy() for i in range(group.shape[0])])
                                       
                        return np.stack(vecs), (A,B)

                                           
        def perform_cca(self, n_components = 64, pca_componenets = 900):
        
                # Perform pca
                
                print("Perorming PCA with {} components on {} vectors...".format(pca_componenets, len(self.data)))
               
                pca = decomposition.PCA(n_components = pca_componenets)
                pca.fit(self.data)
                self.view1, self.view2 = pca.transform(self.view1), pca.transform(self.view2) 
                
                # Perform cca
                
                print("Perorming CCA with {} components on {} vectors...".format(n_components, len(self.view1)))
                
                cca = CCA(n_components = n_components, max_iter = 500000)
                cca.fit(self.view1, self.view2)
                X_transformed, Y_transformed = cca.transform(self.view1, self.view2)
                
                with open("trained_cca.500pts.900pca.128cca", "wb") as f:
                        pickle.dump(cca, f)
                        
                with open("trained_pca.500pts.900", "wb") as f:
                        pickle.dump(pca, f)
                        
                print(cca.x_weights_[:8,:8])
                print("---------------------")
                print(cca.y_weights_[:8,:8])
                return
                                    
        def read_one_set(self, line):
  
                all_vecs_and_sents = line.strip().split("\t")
                
                zipped = [] # tuples of (list_of_vecs, sent)
                
                for i in range(0, len(all_vecs_and_sents) - 1, 2):
                
                        zipped.append((all_vecs_and_sents[i], all_vecs_and_sents[i + 1]))

                all_vecs = []
                all_sents = []
                
                for vecs, sent_str in zipped:
                
                        vecs_str_lst = vecs.split("*")
                        
                        vecs = np.array([np.array([float(x) for x in v.split(" ")]) for v in vecs_str_lst])

                        assert len(vecs_str_lst) == len(sent_str.split(" "))
                        
                        all_vecs.append(vecs)
                        all_sents.append(sent_str)

                sent_length = len(all_sents[0].split(" "))  
                all_vecs = np.stack(all_vecs)
                
                assert all_vecs.shape[0] == len(all_sents)
                assert all_vecs.shape[1] == sent_length
                assert all_vecs.shape[2] == 1024
                
                return all_vecs, all_sents