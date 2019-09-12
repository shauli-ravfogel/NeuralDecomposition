from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of, prepare_global_logging
from allennlp.common.checks import ConfigurationError
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
from allennlp.commands.subcommand import Subcommand
from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField

import logging
import argparse
import json
import logging
import os
from typing import IO, List, Iterable, Tuple, Dict, Union
import warnings
import h5py
from overrides import overrides

import numpy
import torch


DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64


logger = logging.getLogger(__name__)


def empty_embedding() -> numpy.ndarray:
    return numpy.zeros((3, 0, 1024))


class RandomElmoEmbedder:
    def __init__(self,
                 options_file: str = DEFAULT_OPTIONS_FILE,
                 weight_file: str = DEFAULT_WEIGHT_FILE,
                 cuda_device: int = -1,
                 random_emb: bool = True,
                 random_lstm: bool = True) -> None:
        """
        Parameters
        ----------
        options_file : ``str``, optional
            A path or URL to an ELMo options file.
        weight_file : ``str``, optional
            A path or URL to an ELMo weights file.
        cuda_device : ``int``, optional, (default=-1)
            The GPU device to run on.
        """
        self.indexer = ELMoTokenCharactersIndexer()

        logger.info("Initializing ELMo.")
        self.elmo_bilm = CustomElmoBiLm(options_file, weight_file, random_emb=random_emb, random_lstm=random_lstm)
        if cuda_device >= 0:
            self.elmo_bilm = self.elmo_bilm.cuda(device=cuda_device)

        self.cuda_device = cuda_device

    def batch_to_embeddings(self, batch: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A tuple of tensors, the first representing activations (batch_size, 3, num_timesteps, 1024) and
        the second a mask (batch_size, num_timesteps).
        """
        character_ids = batch_to_ids(batch)
        if self.cuda_device >= 0:
            character_ids = character_ids.cuda(device=self.cuda_device)

        bilm_output = self.elmo_bilm(character_ids)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        # without_bos_eos is a 3 element list of (activation, mask) tensor pairs,
        # each with size (batch_size, num_timesteps, dim and (batch_size, num_timesteps)
        # respectively.
        without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
                           for layer in layer_activations]
        # Converts a list of pairs (activation, mask) tensors to a single tensor of activations.
        activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
        # The mask is the same for each ELMo vector, so just take the first.
        mask = without_bos_eos[0][1]

        return activations, mask

    def embed_sentence(self, sentence: List[str]) -> numpy.ndarray:
        """
        Computes the ELMo embeddings for a single tokenized sentence.

        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.

        Parameters
        ----------
        sentence : ``List[str]``, required
            A tokenized sentence.

        Returns
        -------
        A tensor containing the ELMo vectors.
        """

        return self.embed_batch([sentence])[0]

    def embed_batch(self, batch: List[List[str]]) -> List[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a batch of tokenized sentences.

        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.

        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.

        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        elmo_embeddings = []

        # Batches with only an empty sentence will throw an exception inside AllenNLP, so we handle this case
        # and return an empty embedding instead.
        if batch == [[]]:
            elmo_embeddings.append(empty_embedding())
        else:
            embeddings, mask = self.batch_to_embeddings(batch)
            for i in range(len(batch)):
                length = int(mask[i, :].sum())
                # Slicing the embedding :0 throws an exception so we need to special case for empty sentences.
                if length == 0:
                    elmo_embeddings.append(empty_embedding())
                else:
                    elmo_embeddings.append(embeddings[i, :, :length, :].detach().cpu().numpy())

        return elmo_embeddings

    def embed_sentences(self,
                        sentences: Iterable[List[str]],
                        batch_size: int = DEFAULT_BATCH_SIZE) -> Iterable[numpy.ndarray]:
        """
        Computes the ELMo embeddings for a iterable of sentences.

        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.

        Parameters
        ----------
        sentences : ``Iterable[List[str]]``, required
            An iterable of tokenized sentences.
        batch_size : ``int``, required
            The number of sentences ELMo should process at once.

        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        for batch in lazy_groups_of(iter(sentences), batch_size):
            yield from self.embed_batch(batch)

    def embed_file(self,
                   input_file: IO,
                   output_file_path: str,
                   output_format: str = "all",
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   forget_sentences: bool = False,
                   use_sentence_keys: bool = False) -> None:
        """
        Computes ELMo embeddings from an input_file where each line contains a sentence tokenized by whitespace.
        The ELMo embeddings are written out in HDF5 format, where each sentence embedding
        is saved in a dataset with the line number in the original file as the key.

        Parameters
        ----------
        input_file : ``IO``, required
            A file with one tokenized sentence per line.
        output_file_path : ``str``, required
            A path to the output hdf5 file.
        output_format : ``str``, optional, (default = "all")
            The embeddings to output.  Must be one of "all", "top", or "average".
        batch_size : ``int``, optional, (default = 64)
            The number of sentences to process in ELMo at one time.
        forget_sentences : ``bool``, optional, (default = False).
            If use_sentence_keys is False, whether or not to include a string
            serialized JSON dictionary that associates sentences with their
            line number (its HDF5 key). The mapping is placed in the
            "sentence_to_index" HDF5 key. This is useful if
            you want to use the embeddings without keeping the original file
            of sentences around.
        use_sentence_keys : ``bool``, optional, (default = False).
            Whether or not to use full sentences as keys. By default,
            the line numbers of the input file are used as ids, which is more robust.
        """

        assert output_format in ["all", "top", "average"]

        # Tokenizes the sentences.
        sentences = [line.strip() for line in input_file]

        blank_lines = [i for (i, line) in enumerate(sentences) if line == ""]
        if blank_lines:
            raise ConfigurationError(f"Your input file contains empty lines at indexes "
                                     f"{blank_lines}. Please remove them.")
        split_sentences = [sentence.split() for sentence in sentences]
        # Uses the sentence index as the key.

        if use_sentence_keys:
            logger.warning("Using sentences as keys can fail if sentences "
                           "contain forward slashes or colons. Use with caution.")
            embedded_sentences = zip(sentences, self.embed_sentences(split_sentences, batch_size))
        else:
            embedded_sentences = ((str(i), x) for i, x in
                                  enumerate(self.embed_sentences(split_sentences, batch_size)))

        sentence_to_index = {}
        logger.info("Processing sentences.")
        with h5py.File(output_file_path, 'w') as fout:
            for key, embeddings in Tqdm.tqdm(embedded_sentences):
                if use_sentence_keys and key in fout.keys():
                    raise ConfigurationError(f"Key already exists in {output_file_path}. "
                                             f"To encode duplicate sentences, do not pass "
                                             f"the --use-sentence-keys flag.")

                if not forget_sentences and not use_sentence_keys:
                    sentence = sentences[int(key)]
                    sentence_to_index[sentence] = key

                if output_format == "all":
                    output = embeddings
                elif output_format == "top":
                    output = embeddings[-1]
                elif output_format == "average":
                    output = numpy.average(embeddings, axis=0)

                fout.create_dataset(
                        str(key),
                        output.shape, dtype='float32',
                        data=output
                )
            if not forget_sentences and not use_sentence_keys:
                sentence_index_dataset = fout.create_dataset(
                        "sentence_to_index",
                        (1,),
                        dtype=h5py.special_dtype(vlen=str))
                sentence_index_dataset[0] = json.dumps(sentence_to_index)

        input_file.close()


class CustomElmoBiLm(torch.nn.Module):
    """
    Run a pre-trained bidirectional language model, outputting the activations at each
    layer for weighting together into an ELMo representation (with
    ``allennlp.modules.seq2seq_encoders.Elmo``).  This is a lower level class, useful
    for advanced uses, but most users should use ``allennlp.modules.seq2seq_encoders.Elmo``
    directly.

    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional, (default = False).
        If True, compute gradient of ELMo parameters for fine tuning.
    vocab_to_cache : ``List[str]``, optional, (default = None).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, _ElmoBiLm expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None,
                 random_emb: bool = True,
                 random_lstm: bool = True) -> None:
        super(CustomElmoBiLm, self).__init__()

        self._token_embedder = CustomElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad,
                                                          random_weights=random_emb)

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning("You are fine tuning ELMo and caching char CNN word vectors. "
                            "This behaviour is not guaranteed to be well defined, particularly. "
                            "if not all of your inputs will occur in the vocabulary cache.")
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        if vocab_to_cache:
            logging.info("Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
            # They are set in the method so they can be accessed from outside the
            # constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
                                   hidden_size=options['lstm']['projection_dim'],
                                   cell_size=options['lstm']['dim'],
                                   num_layers=options['lstm']['n_layers'],
                                   memory_cell_clip_value=options['lstm']['cell_clip'],
                                   state_projection_clip_value=options['lstm']['proj_clip'],
                                   requires_grad=requires_grad)

        if not random_lstm:
            self._elmo_lstm.load_weights(weight_file)

        # Number of representation layers including context independent layer
        self.num_layers = options['lstm']['n_layers'] + 1

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape ``(batch_size, timesteps)``,
            which represent word ids which have been pre-cached.

        Returns
        -------
        Dict with keys:

        ``'activations'``: ``List[torch.Tensor]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps + 2, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.

        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos = (word_inputs > 0).long()
                # The character cnn part is cached - just look it up.
                embedded_inputs = self._word_embedding(word_inputs) # type: ignore
                # shape (batch_size, timesteps + 2, embedding_dim)
                type_representation, mask = add_sentence_boundary_token_ids(
                        embedded_inputs,
                        mask_without_bos_eos,
                        self._bos_embedding,
                        self._eos_embedding
                )
            except RuntimeError:
                # Back off to running the character convolutions,
                # as we might not have the words in the cache.
                token_embedding = self._token_embedder(inputs)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            token_embedding = self._token_embedder(inputs)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        # Prepare the output.  The first layer is duplicated.
        # Because of minor differences in how masking is applied depending
        # on whether the char cnn layers are cached, we'll be defensive and
        # multiply by the mask here. It's not strictly necessary, as the
        # mask passed on is correct, but the values in the padded areas
        # of the char cnn representations can change.
        output_tensors = [
                torch.cat([type_representation, type_representation], dim=-1) * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
                'activations': output_tensors,
                'mask': mask,
        }

    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
        """
        Given a list of tokens, this method precomputes word representations
        by running just the character convolutions and highway layers of elmo,
        essentially creating uncontextual word vectors. On subsequent forward passes,
        the word ids are looked up from an embedding, rather than being computed on
        the fly via the CNN encoder.

        This function sets 3 attributes:

        _word_embedding : ``torch.Tensor``
            The word embedding for each word in the tokens passed to this method.
        _bos_embedding : ``torch.Tensor``
            The embedding for the BOS token.
        _eos_embedding : ``torch.Tensor``
            The embedding for the EOS token.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            A list of tokens to precompute character convolutions for.
        """
        tokens = [ELMoCharacterMapper.bos_token, ELMoCharacterMapper.eos_token] + tokens
        timesteps = 32
        batch_size = 32
        chunked_tokens = lazy_groups_of(iter(tokens), timesteps)

        all_embeddings = []
        device = get_device_of(next(self.parameters()))
        for batch in lazy_groups_of(chunked_tokens, batch_size):
            # Shape (batch_size, timesteps, 50)
            batched_tensor = batch_to_ids(batch)
            # NOTE: This device check is for when a user calls this method having
            # already placed the model on a device. If this is called in the
            # constructor, it will probably happen on the CPU. This isn't too bad,
            # because it's only a few convolutions and will likely be very fast.
            if device >= 0:
                batched_tensor = batched_tensor.cuda(device)
            output = self._token_embedder(batched_tensor)
            token_embedding = output["token_embedding"]
            mask = output["mask"]
            token_embedding, _ = remove_sentence_boundaries(token_embedding, mask)
            all_embeddings.append(token_embedding.view(-1, token_embedding.size(-1)))
        full_embedding = torch.cat(all_embeddings, 0)

        # We might have some trailing embeddings from padding in the batch, so
        # we clip the embedding and lookup to the right size.
        full_embedding = full_embedding[:len(tokens), :]
        embedding = full_embedding[2:len(tokens), :]
        vocab_size, embedding_dim = list(embedding.size())

        from allennlp.modules.token_embedders import Embedding # type: ignore
        self._bos_embedding = full_embedding[0, :]
        self._eos_embedding = full_embedding[1, :]
        self._word_embedding = Embedding(vocab_size, # type: ignore
                                         embedding_dim,
                                         weight=embedding.data,
                                         trainable=self._requires_grad,
                                         padding_index=0)


class CustomElmoCharacterEncoder(torch.nn.Module):
    """
    Compute context insensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.

    Note: this is a lower level class useful for advanced usage.  Most users should
    use ``ElmoTokenEmbedder`` or ``allennlp.modules.Elmo`` instead.

    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional, (default = False).
        If True, compute gradient of ELMo parameters for fine tuning.

    The relevant section of the options file is something like:
    .. example-code::

        .. code-block:: python

            {'char_cnn': {
                'activation': 'relu',
                'embedding': {'dim': 4},
                'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                'max_characters_per_token': 50,
                'n_characters': 262,
                'n_highway': 2
                }
            }
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 random_weights=True) -> None:
        super(CustomElmoCharacterEncoder, self).__init__()

        with open(cached_path(options_file), 'r') as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file

        self.output_dim = self._options['lstm']['projection_dim']
        self.requires_grad = requires_grad

        self._load_weights(random_weights)

        # Cache the arrays for use in forward -- +1 due to masking.
        self._beginning_of_sentence_characters = torch.from_numpy(
                numpy.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
                numpy.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )

    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.

        Returns
        -------
        Dict with keys:
        ``'token_embedding'``: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2)`` long tensor with sequence mask.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
                inputs,
                mask,
                self._beginning_of_sentence_characters,
                self._end_of_sentence_characters
        )

        # the character id embedding
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
                character_ids_with_bos_eos.view(-1, max_chars_per_token),
                self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'char_conv_{}'.format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
                'mask': mask_with_bos_eos,
                'token_embedding': token_embedding.view(batch_size, sequence_length, -1)
        }

    def _load_weights(self, random):
        self._load_char_embedding(random)
        self._load_cnn_weights(random)
        self._load_highway(random)
        self._load_projection(random)

    def _load_char_embedding(self, random):
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            char_embed_weights = fin['char_embed'][...]

        weights = numpy.zeros(
                (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                dtype='float32'
        )
        weights[1:, :] = char_embed_weights

        self._char_embedding_weights = torch.nn.Parameter(
                torch.FloatTensor(weights), requires_grad=self.requires_grad
        )
        if random:
            torch.nn.init.xavier_uniform_(self._char_embedding_weights.weight)

    def _load_cnn_weights(self, random):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        char_embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True
            )
            # load the weights
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                weight = fin['CNN']['W_cnn_{}'.format(i)][...]
                bias = fin['CNN']['b_cnn_{}'.format(i)][...]

            w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError("Invalid weight file")
            conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
            conv.bias.data.copy_(torch.FloatTensor(bias))

            if random:
                torch.nn.init.xavier_uniform_(conv.weight)
                torch.nn.init.xavier_uniform_(conv.bias)

            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad

            convolutions.append(conv)
            self.add_module('char_conv_{}'.format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self, random):
        # pylint: disable=protected-access
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options['n_highway']

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                # The weights are transposed due to multiplication order assumptions in tf
                # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                w_transform = numpy.transpose(fin['CNN_high_{}'.format(k)]['W_transform'][...])
                # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                w_carry = -1.0 * numpy.transpose(fin['CNN_high_{}'.format(k)]['W_carry'][...])
                weight = numpy.concatenate([w_transform, w_carry], axis=0)
                self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                self._highways._layers[k].weight.requires_grad = self.requires_grad

                b_transform = fin['CNN_high_{}'.format(k)]['b_transform'][...]
                b_carry = -1.0 * fin['CNN_high_{}'.format(k)]['b_carry'][...]
                bias = numpy.concatenate([b_transform, b_carry], axis=0)
                self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
                self._highways._layers[k].bias.requires_grad = self.requires_grad

                if random:
                    torch.nn.init.xavier_uniform_(self._highways._layers[k].weight)
                    torch.nn.init.xavier_uniform_(self._highways._layers[k].bias)

    def _load_projection(self, random):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]
            self._projection.weight.data.copy_(torch.FloatTensor(numpy.transpose(weight)))
            self._projection.bias.data.copy_(torch.FloatTensor(bias))

            if random:
                torch.nn.init.xavier_uniform_(self._projection.weight)
                torch.nn.init.xavier_uniform_(self._projection.bias)

            self._projection.weight.requires_grad = self.requires_grad
            self._projection.bias.requires_grad = self.requires_grad

