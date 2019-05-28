import logging
from typing import Dict, List

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ArrayField
from allennlp.data.instance import Instance
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("reader")
class DataReader(DatasetReader):
    """

    Parameters
    ----------

    Returns
    -------
    A ``Dataset`` of ``Instances`` for NFH identification and resolution.
    """
    def __init__(self,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), 'r') as f:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                vec1 = np.array(list(map(float, parts[3].split())))
                vec2 = np.array(list(map(float, parts[4].split())))
                label = int(parts[5])

                yield self.text_to_instance(vec1, vec2, label)

    @overrides
    def text_to_instance(self, vec1: List, vec2: List, label: int = None) -> Instance:

        fields: Dict[str, Field] = {}

        vec1 = ArrayField(vec1)
        vec2 = ArrayField(vec2)
        fields['vec1'] = vec1
        fields['vec2'] = vec2

        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=True)

        return Instance(fields)
