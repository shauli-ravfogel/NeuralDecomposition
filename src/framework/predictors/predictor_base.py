from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor


@Predictor.register('decomposition_predictor')
class NfhDetectorPredictor(Predictor):
    """"Predictor wrapper for the NfhDetector"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        vec1 = json_dict['vec1']
        vec2 = json_dict['vec2']
        label = json_dict['label'] if 'label' in json_dict else None

        instance = self._dataset_reader.text_to_instance(vec1=vec1, vec2=vec2, label=label)

        return instance
