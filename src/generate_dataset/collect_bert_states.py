import argparse
import numpy as np
# import sys
import pickle
from typing import List
import utils
import h5py

from pytorch_pretrained_bert.modeling import BertConfig, BertModel

from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder

from tqdm import tqdm


import torch
import torch.nn.functional as F
from allennlp.nn import util


# sys.path.append("../../../src/generate_dataset")
FUNCTION_WORDS = utils.DEFAULT_PARAMS["function_words"]


class BertLayerEmbedder(BertEmbedder):
    """
    A ``TokenEmbedder`` that produces BERT embeddings for your tokens.
    Should be paired with a ``BertIndexer``, which produces wordpiece ids.
    Most likely you probably want to use ``PretrainedBertEmbedder``
    for one of the named pretrained models, not this base class.
    Parameters
    ----------
    bert_model: ``BertModel``
        The BERT model being wrapped.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    max_pieces : int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Assuming the inputs are windowed
        and padded appropriately by this length, the embedder will split them into a
        large batch, feed them into BERT, and recombine the output as if it was a
        longer sequence.
    num_start_tokens : int, optional (default: 1)
        The number of starting special tokens input to BERT (usually 1, i.e., [CLS])
    num_end_tokens : int, optional (default: 1)
        The number of ending tokens input to BERT (usually 1, i.e., [SEP])
    """
    def __init__(self,
                 bert_model: BertModel,
                 max_pieces: int = 512,
                 num_start_tokens: int = 1,
                 num_end_tokens: int = 1) -> None:
        super().__init__(bert_model, True, max_pieces, num_start_tokens, num_end_tokens)

    def forward(self,
                input_ids: torch.LongTensor,
                offsets: torch.LongTensor = None,
                token_type_ids: torch.LongTensor = None,
                layer_id: int =-1) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : ``torch.LongTensor``
            The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
        offsets : ``torch.LongTensor``, optional
            The BERT embeddings are one per wordpiece. However it's possible/likely
            you might want one per original token. In that case, ``offsets``
            represents the indices of the desired wordpiece for each original token.
            Depending on how your token indexer is configured, this could be the
            position of the last wordpiece for each token, or it could be the position
            of the first wordpiece for each token.
            For example, if you had the sentence "Definitely not", and if the corresponding
            wordpieces were ["Def", "##in", "##ite", "##ly", "not"], then the input_ids
            would be 5 wordpiece ids, and the "last wordpiece" offsets would be [3, 4].
            If offsets are provided, the returned tensor will contain only the wordpiece
            embeddings at those positions, and (in particular) will contain one embedding
            per token. If offsets are not provided, the entire tensor of wordpiece embeddings
            will be returned.
        token_type_ids : ``torch.LongTensor``, optional
            If an input consists of two sentences (as in the BERT paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            (the default BertIndexer doesn't) then it's assumed to be all 0s.
        """
        # pylint: disable=arguments-differ
        batch_size, full_seq_len = input_ids.size(0), input_ids.size(-1)
        initial_dims = list(input_ids.shape[:-1])

        # The embedder may receive an input tensor that has a sequence length longer than can
        # be fit. In that case, we should expect the wordpiece indexer to create padded windows
        # of length `self.max_pieces` for us, and have them concatenated into one long sequence.
        # E.g., "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ..."
        # We can then split the sequence into sub-sequences of that length, and concatenate them
        # along the batch dimension so we effectively have one huge batch of partial sentences.
        # This can then be fed into BERT without any sentence length issues. Keep in mind
        # that the memory consumption can dramatically increase for large batches with extremely
        # long sentences.
        needs_split = full_seq_len > self.max_pieces
        last_window_size = 0
        if needs_split:
            # Split the flattened list by the window size, `max_pieces`
            split_input_ids = list(input_ids.split(self.max_pieces, dim=-1))

            # We want all sequences to be the same length, so pad the last sequence
            last_window_size = split_input_ids[-1].size(-1)
            padding_amount = self.max_pieces - last_window_size
            split_input_ids[-1] = F.pad(split_input_ids[-1], pad=[0, padding_amount], value=0)

            # Now combine the sequences along the batch dimension
            input_ids = torch.cat(split_input_ids, dim=0)

            if token_type_ids is not None:
                # Same for token_type_ids
                split_token_type_ids = list(token_type_ids.split(self.max_pieces, dim=-1))

                last_window_size = split_token_type_ids[-1].size(-1)
                padding_amount = self.max_pieces - last_window_size
                split_token_type_ids[-1] = F.pad(split_token_type_ids[-1], pad=[0, padding_amount], value=0)

                token_type_ids = torch.cat(split_token_type_ids, dim=0)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        # input_ids may have extra dimensions, so we reshape down to 2-d
        # before calling the BERT model and then reshape back at the end.
        all_encoder_layers, _ = self.bert_model(input_ids=util.combine_initial_dims(input_ids),
                                                token_type_ids=util.combine_initial_dims(token_type_ids),
                                                attention_mask=util.combine_initial_dims(input_mask))
        all_encoder_layers = torch.stack(all_encoder_layers)

        if needs_split:
            # First, unpack the output embeddings into one long sequence again
            unpacked_embeddings = torch.split(all_encoder_layers, batch_size, dim=1)
            unpacked_embeddings = torch.cat(unpacked_embeddings, dim=2)

            # Next, select indices of the sequence such that it will result in embeddings representing the original
            # sentence. To capture maximal context, the indices will be the middle part of each embedded window
            # sub-sequence (plus any leftover start and final edge windows), e.g.,
            #  0     1 2    3  4   5    6    7     8     9   10   11   12    13 14  15
            # "[CLS] I went to the very fine [SEP] [CLS] the very fine store to eat [SEP]"
            # with max_pieces = 8 should produce max context indices [2, 3, 4, 10, 11, 12] with additional start
            # and final windows with indices [0, 1] and [14, 15] respectively.

            # Find the stride as half the max pieces, ignoring the special start and end tokens
            # Calculate an offset to extract the centermost embeddings of each window
            stride = (self.max_pieces - self.num_start_tokens - self.num_end_tokens) // 2
            stride_offset = stride // 2 + self.num_start_tokens

            first_window = list(range(stride_offset))

            max_context_windows = [i for i in range(full_seq_len)
                                   if stride_offset - 1 < i % self.max_pieces < stride_offset + stride]

            # Lookback what's left, unless it's the whole self.max_pieces window
            if full_seq_len % self.max_pieces == 0:
                lookback = self.max_pieces
            else:
                lookback = full_seq_len % self.max_pieces

            final_window_start = full_seq_len - lookback + stride_offset + stride
            final_window = list(range(final_window_start, full_seq_len))

            select_indices = first_window + max_context_windows + final_window

            initial_dims.append(len(select_indices))

            recombined_embeddings = unpacked_embeddings[:, :, select_indices]
        else:
            recombined_embeddings = all_encoder_layers

        # Recombine the outputs of all layers
        # (layers, batch_size * d1 * ... * dn, sequence_length, embedding_dim)
        # recombined = torch.cat(combined, dim=2)
        input_mask = (recombined_embeddings != 0).long()

        if self._scalar_mix is not None:
            mix = self._scalar_mix(recombined_embeddings, input_mask)
        else:
            mix = recombined_embeddings[layer_id]

        # At this point, mix is (batch_size * d1 * ... * dn, sequence_length, embedding_dim)

        if offsets is None:
            # Resize to (batch_size, d1, ..., dn, sequence_length, embedding_dim)
            dims = initial_dims if needs_split else input_ids.size()
            return util.uncombine_initial_dims(mix, dims)
        else:
            # offsets is (batch_size, d1, ..., dn, orig_sequence_length)
            offsets2d = util.combine_initial_dims(offsets)
            # now offsets is (batch_size * d1 * ... * dn, orig_sequence_length)
            range_vector = util.get_range_vector(offsets2d.size(0),
                                                 device=util.get_device_of(mix)).unsqueeze(1)
            # selected embeddings is also (batch_size * d1 * ... * dn, orig_sequence_length)
            selected_embeddings = mix[range_vector, offsets2d]

            return util.uncombine_initial_dims(selected_embeddings, offsets.size())


def get_equivalent_sentences(equivalent_sentences_path: str, num_sentences: int) -> List[List[List[str]]]:
    # equivalent_sentences_path is the path to a file containing 150k groups of equivalent sentences.
    # each group contains k=15 sentences, represented as a list of lists of string.
    # e.g., if the length of the sentences in the first group is L=20,
    # then sentences[0] is a KxL=15x20 list, where position i,j contains the jth word in the ith sentence.

    with open(equivalent_sentences_path, "rb") as f:
        sentences = pickle.load(f)  # a list of groups. each group is a list of lists of strings
        sentences = list(sentences.values())

    return sentences[:num_sentences]


def get_bert_states(sentence_group: List[List[str]], embedder, layer: int):
    instances = []
    for sen in sentence_group:
        toks = [Token(w) for w in sen]

        instance = Instance({"tokens": TextField(toks, {"bert": token_indexer})})
        instances.append(instance)

    batch = Batch(instances)
    batch.index_instances(vocab)

    padding_lengths = batch.get_padding_lengths()
    tensor_dict = batch.as_tensor_dict(padding_lengths)
    tokens = tensor_dict["tokens"]

    bert_vectors = embedder(tokens["bert"], offsets=tokens["bert-offsets"], layer_id=layer)

    return bert_vectors.data.numpy()


def save_bert_states(embedder, equivalent_sentences: List[List[List[str]]], output_file: str,
                     layer: int):

    with h5py.File(output_file, 'w') as h5:
        for i, group_of_equivalent_sentences in tqdm(enumerate(equivalent_sentences)):
            bert_states = get_bert_states(group_of_equivalent_sentences, embedder, layer)
            # if the length (num of words) of the group i is L, and there are K=15 sentences in the group,
            # then bert_states is a numpy array of dims KxLxD where D is the size of the bert vectors.

            L = len(group_of_equivalent_sentences[0])  # group's sentence length
            content_indices = np.array([i for i in range(L) if group_of_equivalent_sentences[0][i] not in FUNCTION_WORDS])

            sents = np.array(group_of_equivalent_sentences, dtype=object)

            # data.append(bert_states)
            g = h5.create_group(str(i))
            g.attrs['group_size'], g.attrs['sent_length'] = sents.shape
            g.create_dataset('vecs', data=bert_states, compression=True, chunks=True)
            dt = h5py.special_dtype(vlen=str)
            g.create_dataset('sents', data=sents, dtype=dt, compression=True, chunks=True)
            g.create_dataset('content_indices', data=content_indices, compression=True, chunks=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Equivalent sentences generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-sentences', dest='input_sentences', type=str,
                        default='data/interim/bert_online_sents_same_pos4.pickle',
                        help='equivalent sentences to parse with bert')
    parser.add_argument('--bert-model', dest='bert_model', type=str,
                        default='bert-base-uncased',
                        help='bert model type to use. bert-base-uncased / bert-large-uncased / ...')
    parser.add_argument('--output-file', dest='output_file', type=str,
                        default='data/interim/encoder_bert/sents_bert.hdf5',
                        help='output file where the encoded vectors are stored')
    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=30522,
                        help='The size of bert\'s vocabulary')
    parser.add_argument('--num-sentences', dest='num_sentences', type=int, default=999999999,
                        help='The amount of group sentences to use')
    parser.add_argument('--layer', dest='layer', type=int, default=-1,
                        help='The layer of bert to persist')

    args = parser.parse_args()
    all_groups = get_equivalent_sentences(args.input_sentences, args.num_sentences)

    config = BertConfig(vocab_size_or_config_json_file=args.vocab_size)
    bert_model = BertModel.from_pretrained(args.bert_model)

    token_indexer = PretrainedBertIndexer(pretrained_model=args.bert_model, use_starting_offsets=True)
    vocab = Vocabulary()
    tlo_embedder = BertLayerEmbedder(bert_model).eval()

    save_bert_states(tlo_embedder, all_groups, args.output_file, args.layer)
