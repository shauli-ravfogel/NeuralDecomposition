import json
import logging
from typing import Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SpanField, LabelField, IndexField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

import spacy

logger = logging.getLogger(__name__)


@DatasetReader.register("dep_reader")
class NFHReader(DatasetReader):
    """
    This DatasetReader is designed to read a csv file of the nfh dataset.

    It returns a dataset of instances with the following fields:
    sentence : ``TextField``
        The tokens in the sentence.
    anchor_span : ``SpanField``
        The span of the anchor (number)
    label : ``LabelField``
        The label of the instance
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    Returns
    -------
    A ``Dataset`` of ``Instances`` for NFH identification and resolution.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.nlp = spacy.load('en_core_web_sm')

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), 'r') as f:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in f:
                line = line.strip()
                if not line:
                    continue

                doc = self.nlp(line)
                tokens = [x.text for x in doc]
                for w in doc:
                    yield self.text_to_instance(tokens, w.i, w.dep_, w.head.i)

    @overrides
    def text_to_instance(self, tokens: List[str], index: int, dep_edge: str = None, dep_head: int = None) -> Instance:
        # getting the scene from each group and reindexing inner index in every group

        fields: Dict[str, Field] = {}

        sentence = TextField([Token(t) for t in tokens], self._token_indexers)
        fields['sentence'] = sentence
        fields['index'] = IndexField(index, sentence)
        metadata = {"tokens": tokens}
        fields['metadata'] = MetadataField(metadata)

        if dep_edge is not None:
            dep_label = LabelField(dep_edge, label_namespace='dep_label', skip_indexing=False)
            fields['dep_label'] = dep_label

        if dep_head is not None:
            dep_head_label = IndexField(dep_head, sentence)
            fields['dep_head'] = dep_head_label

        return Instance(fields)

