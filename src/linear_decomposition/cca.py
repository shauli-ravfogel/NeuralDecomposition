from sklearn import cross_decomposition, decomposition
import numpy_cca
import numpy as np
import pickle

def run_cca(views_path, perform_pca, pca_dim, cca_dim, enforce_symmetry, model):

        # load views
        
        with open(views_path, "rb") as f:
        
                views = pickle.load(f)
        
        view1, view2, positions = map(np.squeeze, map(np.asarray, zip(*views))) # each view is a numpy arrays
        view1, view2 = np.array(view1), np.array(view2)

        # enforce symmetry
        
        if enforce_symmetry:
        
                view1 += view2
                view2 += view1
        
        # perform pca
              
        if perform_pca:
                
                print("Performing PCA on {} vectors to dimensionality {}".format(len(view1) + len(view2), pca_dim))
                pca = decomposition.PCA(n_components = pca_dim)
                pca.fit(np.concatenate((view1, view2)))
                view1 = pca.inverse_transform(pca.transform(view1))
                view2 = pca.inverse_transform(pca.transform(view2))
       
       # perform cca

        print("Performing CCA on {} vector pairs to dimensionality {}".format(view1.shape[0], cca_dim))
                       
        if model == "numpy":
   
                cca = numpy_cca.CCAModel(cca_dim)
                cca(view1, view2)
                corrs = cca.D[-cca_dim:]

        elif model == "sklearn":   
       
                cca = cross_decomposition.CCA(n_components = cca_dim, max_iter = 500000, tol = 1e-6)
                cca.fit(view1, view2)
                x_proj, y_proj = cca.transform(view1, view2)
                corrs = get_sklearn_cca_corr(x_proj, y_proj)   
       
        print("Correlations: {}; Avergage correlation: {}".format(corrs, np.mean(corrs)))  
        return cca


def get_sklearn_cca_corr(X, Y):

        corrs = [np.corrcoef(X[:,i], Y[:, i])[0,1] for i in range(X.shape[1])]
        return corrs
                    
