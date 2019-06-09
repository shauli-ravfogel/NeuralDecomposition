import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
from allennlp.training.trainer import Trainer
from framework.dataset_readers.data_reader import DataReader
from framework.models.siamese_norm import SiameseModel

torch.manual_seed(1)

reader = DataReader()
# augmented
# sample
dir_path = '/home/lazary/workspace/thesis/tree-extractor/data/'
train_dataset = reader.read(cached_path(dir_path + 'small_train'))
validation_dataset = reader.read(cached_path(dir_path + 'small_dev'))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
scorer = FeedForward(1024, num_layers=2,
                     hidden_dims=[150, 2], activations=[Activation.by_name('tanh')(),
                                                        Activation.by_name('linear')()],
                     dropout=0.2)
representer = FeedForward(1024, num_layers=2,
                          hidden_dims=[512, 1024], activations=[Activation.by_name('tanh')(),
                                                                Activation.by_name('linear')()],
                          dropout=0.2)
model = SiameseModel(representer)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
iterator = BasicIterator(batch_size=2)
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()
