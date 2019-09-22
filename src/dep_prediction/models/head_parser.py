from typing import Dict, Optional, List, Any

from overrides import overrides

import numpy as np
import torch
from torch.nn.modules.linear import Linear
from nltk import Tree

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.similarity_functions import SimilarityFunction, BilinearSimilarity
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import masked_softmax, get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import EvalbBracketingScorer, DEFAULT_EVALB_DIR
from allennlp.common.checks import ConfigurationError
from allennlp.commands.elmo import ElmoEmbedder

from analysis.triplet_extractor import TripletExtractor


@Model.register("head_model")
class DependencyModel(Model):
    """
    This ``SpanConstituencyParser`` simply encodes a sequence of text
    with a stacked ``Seq2SeqEncoder``, extracts span representations using a
    ``SpanExtractor``, and then predicts a label for each span in the sequence.
    These labels are non-terminal nodes in a constituency parse tree, which we then
    greedily reconstruct.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    span_extractor : ``SpanExtractor``, required.
        The method used to extract the spans from the encoded sequence.
    encoder : ``Seq2SeqEncoder``, required.
        The encoder that we will use in between embedding tokens and
        generating span representations.
    feedforward : ``FeedForward``, required.
        The FeedForward layer that we will use in between the encoder and the linear
        projection to a distribution over span labels.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    evalb_directory_path : ``str``, optional (default=``DEFAULT_EVALB_DIR``)
        The path to the directory containing the EVALB executable used to score
        bracketed parses. By default, will use the EVALB included with allennlp,
        which is located at allennlp/tools/EVALB . If ``None``, EVALB scoring
        is not used.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 syntactic_extractor_path: str = None) -> None:
        super().__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.bilinear = BilinearSimilarity(300, 300)

        if syntactic_extractor_path is not None:
            self.syntax_extractor = TripletExtractor(syntactic_extractor_path)
        else:
            self.syntax_extractor = None

        self.use_raw_tokens = False
        if self.use_raw_tokens:
            options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)

        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {'accuracy': CategoricalAccuracy()}


    def fill_blanks(self, encoded_sents):
        vecs = []
        for i in range(len(encoded_sents)):
            sent_embs = np.concatenate([encoded_sents[i][layer] for layer in [1, 2]], axis=1)
            vecs.append(torch.tensor(sent_embs))

        return torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)

    @overrides
    def forward(self,  # type: ignore
                sentence: Dict[str, torch.LongTensor],
                index: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                dep_head: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentence : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        indices : ``torch.LongTensor``, required.


        dep_labels : ``torch.LongTensor``, optional (default = None)
            A torch tensor representing the integer gold class labels for all possible
        dep_heads : ``torch.LongTensor``, optional (default = None)

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        if self.use_raw_tokens:
            raw_tokens = [meta["tokens"] for meta in metadata]
            res = self.elmo.embed_batch(raw_tokens)
            embedded_text_input = self.fill_blanks(res)
            embedded_text_input = embedded_text_input.cuda()
        else:
            embedded_text_input = self.text_field_embedder(sentence)

        if self.syntax_extractor is not None:
            embedded_text_input = self.syntax_extractor.extract_syntax(embedded_text_input)

        selected_indices = embedded_text_input.gather(1,
                                                      index.view(-1, 1)
                                                      .unsqueeze(2).repeat(1, 1, embedded_text_input.shape[-1])).squeeze(1)

        logits = self.bilinear(embedded_text_input, selected_indices.unsqueeze(1))

        y_hat = torch.argmax(logits, dim=1)

        output = {"tag_logits": logits, "y_hat": y_hat}

        if dep_head is not None:
            self.metrics['accuracy'](logits, dep_head.squeeze(1))

            loss = self.loss(logits, dep_head.squeeze(1))
            output["loss"] = loss

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.metrics['accuracy'].get_metric(reset=reset)}
