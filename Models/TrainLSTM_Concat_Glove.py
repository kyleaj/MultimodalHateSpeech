from Trainer import Trainer
from DataLoader import ImFeatureDataLoader_Glove
from LSTM_Concat import LSTM_Concat
import torch
import os
import sys

glove_dim = sys.argv[1]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU!")
else:
    device = torch.device("cpu")
    print("Using CPU :(")

embed_dir = "glove.twitter.27B." + glove_dim + "d.txt"

train_data = ImFeatureDataLoader_Glove("dev.jsonl", "Resnet152", device, embed_dir)
val_data = ImFeatureDataLoader_Glove("dev.jsonl", "Resnet152", device, embed_dir, embedding_dict=train_data.embedding_dict)

model = LSTM_Concat(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss)
trainer.train()