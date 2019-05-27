import torch

import numpy as np
from torch import nn
from torch import optim
import tqdm
from torch.utils import data

import matplotlib.pyplot as plt


class SiameseNet(nn.Module):
	def __init__(self):
		super(SiameseNet, self).__init__()

		input_representation = []
		input_representation.append(nn.Linear(1024, 512))
		#input_representation.append(nn.LeakyReLU(0.2))
		#layers.append(nn.Dropout(0.1))
		input_representation.append(nn.Linear(512, 512))
		#input_representation.append(nn.LeakyReLU(0.2))
		#layers.append(nn.Dropout(0.1))
		input_representation.append(nn.Linear(512, 1024))    
		#input_representation.append(nn.LeakyReLU(0.2))
		#layers.append(nn.Dropout(0.3))
		input_representation.append(nn.Linear(1024, 1024))
		self.representation_net = nn.Sequential(*input_representation)
        
		layers = []
		layers.append(nn.Linear(1024, 512))
		layers.append(nn.LeakyReLU(0.2))
		layers.append(nn.Linear(512, 512))
		layers.append(nn.LeakyReLU(0.2))
		layers.append(nn.Linear(512, 256))
		layers.append(nn.LeakyReLU(0.2))
		layers.append(nn.Linear(256, 2))
		self.prediction_net = nn.Sequential(*layers)
	
	def _represent(self, x):
	
		return self.representation_net(x)
		
	def forward(self, x1, x2):

		h1 = self._represent(x1)
		h2 = self._represent(x2)
		
		h = torch.abs(h1 - h2)
		prediction = self.prediction_net(h)
		
		return prediction

def from_string(vec_str):

	return np.array([float(x) for x in vec_str.split(" ")])
	
def load_data(data):
		
		all_data = []
		
		for i, line in enumerate(data):

			k, sent1, sent2, vec1, vec2, y = line.strip().split("\t")
			sent1, sent2 = sent1.split(" "), sent2.split(" ")
			vec1, vec2 = from_string(vec1), from_string(vec2)
			k = int(k)
			y = int(y)
			all_data.append({"x1": vec1, "x2": vec2, "x": np.concatenate([vec1, vec2]), "y": y, "sent1": sent1, "sent2": sent2, "k": k})
		return all_data
		
			
class Dataset(data.Dataset):

	def __init__(self, data_location):    
		with open(data_location, "r") as f:
			self.lines = f.readlines()
			self.lines = load_data(self.lines)

	def __len__(self):

       		return len(self.lines)

	def __getitem__(self, index):
  
		data_dictionary = self.lines[index]
		x1, x2, y = data_dictionary["x1"], data_dictionary["x2"], data_dictionary["y"]
		return (torch.from_numpy(x1).float(), torch.from_numpy(x2).float()), y
        
def train(model, num_epochs = 6, batch_size = 100):

	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters())
	train = Dataset("data/train")
	dev = Dataset("data/dev")
	training_generator = data.DataLoader(train, batch_size = batch_size, shuffle = True)
	dev_generator = data.DataLoader(dev, batch_size = 1, shuffle = False)
	
	for epoch in range(num_epochs):

		model.train()

		t = tqdm.tqdm(iter(training_generator), leave=False, total=len(training_generator))
		
		for (x1_batch, x2_batch), y_batch in t:
		
			model.zero_grad()
			prediction = model(x1_batch, x2_batch)

			loss = loss_fn(prediction, y_batch)
			loss.backward()
			optimizer.step()
		
		print("Evaluating...")
		evaluate(model, dev_generator)
		print("Epoch {}".format(epoch))

def evaluate(model, eval_generator):

	model.eval()
	good, bad = 0., 0.
	t = tqdm.tqdm(iter(eval_generator), leave = False, total = len(eval_generator))

	with torch.no_grad():

		for (x1, x2), y in t:

			prediction = model(x1, x2)
			y_pred = np.argmax(prediction.detach().numpy())

			if y_pred == y:
				good += 1
			else:
				bad += 1
			
		print(good / (good + bad))

		
	
def main():

	model = SiameseNet()
	train(model)
	torch.save(model.state_dict(), "model.pt")
	
#main()   
     
