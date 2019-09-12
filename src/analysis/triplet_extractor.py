import torch
import pickle

class TripletExtractor(object):

        def __init__(self, model_path):
        
                with open(model_path, "rb") as f:
                
                        self.model = pickle.load(f)

                self.model.eval()
                
        def extract_syntax(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
                """
    
                :param embeddings: ``torch.FloatTensor`` 
                        a torch tensor that have a shape of: (batch_size, sequence length, emb_dim)
                :return: ``torch.FloatTensor``
                a torch tensor that have a shape of: (batch_size, sequence length, emb_dim_reduce)
                """
                with torch.no_grad():
                        _, H = self.model.process(embeddings)
                return H
        
def test_model(model: TripletExtractor):

        batch_size, sent_len, dim = 50, 20, 2048
        x = torch.rand(batch_size, sent_len, dim).cuda()
        h = model.extract_syntax(x)

        assert h.shape == (batch_size, sent_len, 128)
        
model = TripletExtractor("TripletModel_last.pickle")
test_model(model)
