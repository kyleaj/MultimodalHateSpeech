from Trainer import Trainer
from DataLoader import SBERTFeatureDataLoader
from MCB_PreTrainedTransformer import MCB_Late_Fusion
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

train_data = SBERTFeatureDataLoader("train.jsonl", "Resnet152", device, embed_dir)
val_data = SBERTFeatureDataLoader("dev.jsonl", "Resnet152", device, embed_dir)

print("Embed dimension:")
print(train_data.embed_dim)

model = MCB_Late_Fusion(train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss, file_name="MCB_SBERT")
trainer.train()