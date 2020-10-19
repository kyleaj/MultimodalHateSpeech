from Trainer import Trainer
from DataLoader import ImFeatureDataLoader_OneHotEmbed
from LSTM_Concat import LSTM_Concat
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU!")
else:
    device = torch.device("cpu")
    print("Using CPU :(")

train_data = ImFeatureDataLoader_OneHotEmbed("train.jsonl", "Resnet152", device)
val_data = ImFeatureDataLoader_OneHotEmbed("dev.jsonl", "Resnet152", device, 
                vocabulary=train_data.vocabulary, max_len=train_data.max_len)

model = LSTM_Concat(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 512).to(device)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

trainer = Trainer(model, train_data, val_data, opt, loss)
trainer.train()
