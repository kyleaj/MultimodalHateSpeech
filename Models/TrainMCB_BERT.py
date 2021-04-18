from Trainer import Trainer
from DataLoader import BERTFeatureDataLoader
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

train_json = "train.jsonl"
dev_json = "dev.jsonl"

if len(sys.argv) > 1 and sys.argv[1] == "spell":
    train_json = "SpellCheckOuts/" + train_json
    dev_json = "SpellCheckOuts/" + dev_json

train_data = BERTFeatureDataLoader(train_json, "Resnet152", device, embed_dir)
val_data = BERTFeatureDataLoader(dev_json, "Resnet152", device, embed_dir)

print("Embed dimension:")
print(train_data.embed_dim)

model = MCB_Late_Fusion(train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss, file_name="MCB_BERT_")
trainer.train()