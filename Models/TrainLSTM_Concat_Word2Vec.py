from Trainer import Trainer
from DataLoader import ImFeatureDataLoader_Word2Vec
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

train_json = "train.jsonl"
dev_json = "dev.jsonl"

if len(sys.argv) > 1 and sys.argv[1] == "spell":
    train_json = "SpellCheckOuts/" + train_json
    dev_json = "SpellCheckOuts/" + dev_json

train_data = ImFeatureDataLoader_Word2Vec(train_json, "Resnet152", device, embed_dir)
val_data = ImFeatureDataLoader_Word2Vec(dev_json, "Resnet152", device, embed_dir, embedding_dict=train_data.embedding_dict)

model = None

if len(sys.argv) == 2 and sys.argv[1] != "spell":
    print("Loading pretrained model...")
    model = torch.load(sys.argv[1])
    for param in model.LSTM.parameters():
        param.requires_grad = False
    #for param in model.decoder.parameters():
    #    param.requires_grad = False
else:
    model = LSTM_Concat(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss, file_name="Word2Vec")
trainer.train()