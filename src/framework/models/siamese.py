from typing import Dict

import torch
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.training.metrics import BooleanAccuracy
from overrides import overrides


@Model.register("siamese")
class SiameseModel(Model):
    """


    Parameters
    ----------
    """
    def __init__(self,
                 input_rep: FeedForward,
                 scorer: FeedForward):
        super().__init__(None)
        self.rep = input_rep
        self.scorer = scorer

        self.loss = torch.nn.CrossEntropyLoss()
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

        rep1 = self.rep(vec1)
        rep2 = self.rep(vec2)

        h = rep1 - rep2

        scores = self.scorer(h)

        y_hat = torch.argmax(scores, dim=1)

        output = {"tag_logits": scores, "y_hat": y_hat}
        if label is not None:
            self.accuracy(y_hat.reshape(-1), label)
            output["loss"] = self.loss(scores, label)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
