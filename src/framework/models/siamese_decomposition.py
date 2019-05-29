from typing import Dict

import torch
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.training.metrics import BooleanAccuracy
from overrides import overrides


@Model.register("siamese_decomposition")
class SiameseDecompositionModel(Model):
    """


    Parameters
    ----------
    """
    def __init__(self,
                 syntax: FeedForward,
                 semantic: FeedForward,
                 inverse: FeedForward,
                 semantic_predictor: FeedForward,
                 siamese: FeedForward):
        super().__init__(None)
        self.syntax = syntax
        self.semantic = semantic
        self.inverse = inverse
        self.semantic_predictor = semantic_predictor
        self.siamese = siamese

        self.loss = torch.nn.CrossEntropyLoss()
        self.reconstruction_loss = torch.nn.L1Loss()
        self.accuracy = BooleanAccuracy()

    @overrides
    def forward(self,
                vec1: torch.Tensor,
                vec2: torch.Tensor,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------


        Returns
        -------
        An output dictionary consisting of:
        tag_logits: torch.FloatTensor, required
            A tensor of shape ``(batch_size, max_sentence_length)``
            representing a distribution over the label classes for each instance.
        loss: torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        syn1, sem1 = self.decomposition(vec1)
        syn2, sem2 = self.decomposition(vec2)

        h = syn1 - syn2

        scores = self.siamese(h)

        y_hat = torch.argmax(scores, dim=1)

        re_decomposition1 = self.re_decomposition(syn1, sem1)
        re_decomposition2 = self.re_decomposition(syn2, sem2)

        sem_hat1 = self.predict_semantic(syn1)
        sem_hat2 = self.predict_semantic(syn2)

        output = {"tag_logits": scores, "y_hat": y_hat}
        if label is not None:
            self.accuracy(y_hat.reshape(-1), label)
            output["loss"] = self.loss(scores, label)

            output["loss"] += self.reconstruction_loss(vec1, re_decomposition1)
            output["loss"] += self.reconstruction_loss(vec2, re_decomposition2)

            output["loss"] += self.reconstruction_loss(sem_hat1, sem1)
            output["loss"] += self.reconstruction_loss(sem_hat2, sem2)

        return output

    def decomposition(self, vec):
        syn = self.syntax(vec)
        sem = self.semantic(vec)

        return syn, sem

    def re_decomposition(self, syn, sem):
        vec = torch.cat([syn, sem], dim=1)
        origin = self.inverse(vec)
        return origin

    def predict_semantic(self, syn):
        sem = self.semantic_predictor(grad_reverse(syn))
        return sem

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return (-grad_output)


def grad_reverse(x):
    return GradReverse()(x)