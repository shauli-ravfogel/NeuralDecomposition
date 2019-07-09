import numpy as np
import sys
import pickle

class SyntacticExtractor(object):

        def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        
                raise NotImplementedError
                
                
class SiameseSyntacticExtractor(SyntacticExtractor):

        def __init__(self): 
        
                SyntacticExtractor.__init__(self)
                sys.path.append('../src/framework/models')
                sys.path.append('../src/framework/dataset_readers')
                sys.path.append('../src/framework/predictors')
        
                from allennlp.models.archival import load_archive
                from allennlp.predictors import Predictor
                from predictor_base import NfhDetectorPredictor

                archive_elmo = load_archive('../allen_logs/pos_base/model.tar.gz', cuda_device=2)
                predictor_elmo = Predictor.from_archive(archive_elmo, 'decomposition_predictor')                
        
        def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        
                raise NotImplementedError
                
                

class CCASyntacticExtractor(SyntacticExtractor):

        def __init__(self): 
        
                with open("trained_pca.1500pts.900", "rb") as f:
                
                        self.pca = pickle.load(f)

                with open("trained_cca.1500pts.950pca.45cca", "rb") as f:
                
                        self.cca = pickle.load(f)                       
        
        def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        
                x = self.pca.inverse_transform(self.pca.transform(contextualized_vector))
                return self.cca.transform(x)
                

class PCASyntacticExtractor(SyntacticExtractor):

        def __init__(self): 
        
                with open("fitted_pca.init.dim900", "rb") as f:
                
                        self.initial_pca = pickle.load(f)

                with open("fitted_pca.500pts.900.pca-init.100pca", "rb") as f:
                
                        self.pca = pickle.load(f)
        
        def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        
                """
                x = self.initial_pca.transform(contextualized_vector)
                y = x - self.pca.inverse_transform(self.pca.transform(x))
                """
                
                x = self.initial_pca.inverse_transform(self.initial_pca.transform(contextualized_vector))
                y = self.pca.inverse_transform(self.pca.transform(x))
                
                return y
                                
#extractor = SiameseSyntacticExtractor()
