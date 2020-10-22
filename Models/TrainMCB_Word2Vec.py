from Trainer import Trainer
from DataLoader import ImFeatureDataLoader_Word2Vec
from MCB_Late_Fusion import MCB_Late_Fusion
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

train_data = ImFeatureDataLoader_Word2Vec("train.jsonl", "Resnet152", device, embed_dir)
val_data = ImFeatureDataLoader_Word2Vec("dev.jsonl", "Resnet152", device, embed_dir, embedding_dict=train_data.embedding_dict)

model = MCB_Late_Fusion(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss, file_name="Word2Vec")
trainer.train()