import random
random.seed(0)

with open("data.txt", "r") as f:

	data = f.readlines()

random.shuffle(data)
train_prop = 0.8
dev_prop = 0.1
train_size = int(len(data) * train_prop)
dev_size = int(len(data) * dev_prop)

train = data[:train_size]
dev = data[train_size: train_size + dev_size]
test = data[train_size + dev_size:]

parts = [train, dev, test]

for part, name in zip(parts, ["train", "dev", "test"]):


	with open(name, "w") as f:

		for line in part:

			f.write(line)



