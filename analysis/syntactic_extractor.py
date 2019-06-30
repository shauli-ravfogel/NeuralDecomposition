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
	
                with open("trained_pca.500pts.900", "rb") as f:
                
                        self.pca = pickle.load(f)

                with open("trained_cca.500pts.900pca.64cca", "rb") as f:
                
                        self.cca = pickle.load(f)                       
	
	def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
	
		return self.cca.transform(self.pca.transform(contextualized_vector))
		
		
#extractor = SiameseSyntacticExtractor()
