import torch
from torch import nn
from pytorch_revgrad import RevGrad

class Siamese(nn.Module):

    def __init__(self, dim = 2048, final = 512):

        super(Siamese, self).__init__()

        layer_sizes = [dim, 1500, 1024, final]
        layers = []

        for i, (layer_dim, next_layer_dim) in enumerate(zip(layer_sizes,layer_sizes[1:])):

            #layers.append(nn.BatchNorm1d(layer_dim))
            #if i == 0:
            #    layers.append(GaussianNoise(stddev=0.001))
            layers.append(nn.Linear(layer_dim, next_layer_dim, bias = True))
            if i != len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def process_word(self, word_vec):

        return self.layers(word_vec)

    def get_batch_mean(self, batch_transformed, sent_lengths):

        max_len = batch_transformed.shape[1]
        mask = torch.arange(max_len)[None, :].cuda() < sent_lengths[:, None] # mask padded elements

        # zero out the padded elements

        masked = mask[..., None].float().cuda() * batch_transformed

        # calcualte means

        summed = torch.sum(masked, axis = 1) # (BATCH_SIZE x dim)
        mean = summed / sent_lengths[:, None].float() #(BATCH_SIZE, dim)

        return mean

    def forward(self, sent_vecs, lengths):

        transformed =  self.layers(sent_vecs) # (BATCH_SIZE x MAX_SENT_LENGTH x 2048)
        mean = self.get_batch_mean(transformed, lengths)
        return mean

if __name__ == '__main__':

    dataset = dataset.Dataset("sample.hdf5")
    model = Siamese()
    vecs1, vecs2 = dataset[0]
    print(model(vecs1[0]))
