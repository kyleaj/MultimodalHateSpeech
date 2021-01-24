from Trainer import Trainer
from DataLoader import ImFeatureDataLoader_Coco_Word2Vec
from LSTM_Concat import LSTM_Concat
import torch
import os
import sys

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU!")
else:
    device = torch.device("cpu")
    print("Using CPU :(")

embed_dir = "/tigress/kyleaj/Thesis/Embeddings/GoogleNews-vectors-negative300.bin"

train_data = ImFeatureDataLoader_Coco_Word2Vec("/tigress/kyleaj/Thesis/CocoDataset/annotations/captions_train2014.json", "/tigress/kyleaj/Thesis/CocoDataset/train2014", device, embed_dir)

model = LSTM_Concat(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 512, lstm_dropout=0.4).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, None, opt, loss, file_name="Word2Vec")
trainer.train(epochs=100, batch_size=32)