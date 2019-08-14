import numpy as np
import sys
import pickle
import torch


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
        with open("models/PCAModel.PCA:1900.Method:cca.Dim:80.Data:bert_same_pos.Sentences:16000.pickle", "rb") as f:
            self.pca = pickle.load(f)

        with open("models/CCAModel.PCA:1900.Method:cca.Dim:80.Data:bert_same_pos.Sentences:16000.pickle", "rb") as f:
            self.cca = pickle.load(f)

    def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        x = self.pca.inverse_transform(self.pca.transform(
            [contextualized_vector] if len(contextualized_vector.shape) == 1 else contextualized_vector))
        return self.cca.transform(x)[:]


class NeuralCCASyntacticExtractor(SyntacticExtractor):

    def __init__(self):
        import model
        # self.model = model.ProjectionNetwork()
        # self.model.load_state_dict(torch.load("NeuralCCA.pickle"))
        self.model = torch.load("NeuralCCA.pickle")
        self.model.eval()
        self.model.cpu()
        self.model.cca.mean_x.cpu()
        self.model.cca.mean_y.cpu()
        self.model.cca.A.cpu()
        self.model.cca.B.cpu()

    def extract(self, contextualized_vector: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(contextualized_vector).float()
            h = self.model.layers(x.unsqueeze(0) if len(x.shape) == 1 else x)
            return h.detach().numpy()[:, :]

            T, (o, _) = self.model.cca(h.cuda(), h.cuda(), is_training=False)
            corr = torch.diag(T).detach().cpu().numpy()

            # print("---------------------------------------------------------")
            return o.detach().cpu().numpy()[:, :]
