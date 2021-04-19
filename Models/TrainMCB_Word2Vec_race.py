from Trainer import Trainer
from DataLoader import ImFeatureDataLoader_Word2Vec_RaceGender
from MCB_Race import MCB_RaceGender
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

race_dev = "/home/kyleaj/Thesis/code/RaceGenderClassify/face_outs/dev_infilled.csv"
race_train = "/home/kyleaj/Thesis/code/RaceGenderClassify/face_outs/train_infilled.csv"

if len(sys.argv) > 1 and sys.argv[1] == "spell":
    train_json = "SpellCheckOuts/" + train_json
    dev_json = "SpellCheckOuts/" + dev_json

train_data = ImFeatureDataLoader_Word2Vec_RaceGender(train_json, "Resnet152", device, embed_dir, race_gender_path=race_train)
val_data = ImFeatureDataLoader_Word2Vec_RaceGender(dev_json, "Resnet152", device, embed_dir, 
                            race_gender_path=race_dev, embedding_dict=train_data.embedding_dict)

model = MCB_RaceGender(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss, file_name="MCB_Word2Vec_")
trainer.train()