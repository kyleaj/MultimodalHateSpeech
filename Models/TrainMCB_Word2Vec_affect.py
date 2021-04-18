from Trainer import Trainer
from DataLoader import ImFeatureDataLoader_Word2Vec_AffectNet
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

train_json = "train.jsonl"
dev_json = "dev.jsonl"

if len(sys.argv) > 1 and sys.argv[1] == "spell":
    train_json = "SpellCheckOuts/" + train_json
    dev_json = "SpellCheckOuts/" + dev_json

embed_dir = "/tigress/kyleaj/Thesis/Embeddings/GoogleNews-vectors-negative300.bin"
aff_path = "/home/kyleaj/Thesis/code/AffectiveSpace/affectivespace-pkl/affectivespace.pkl"

train_data = ImFeatureDataLoader_Word2Vec_AffectNet(train_json, "Resnet152", device, embed_dir, aff_path)
val_data = ImFeatureDataLoader_Word2Vec_AffectNet(dev_json, "Resnet152", device, embed_dir, aff_path, embedding_dict=train_data.embedding_dict)

model = MCB_Late_Fusion(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss, file_name="affect_MCB_Word2Vec")
trainer.train()