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

train_data = ImFeatureDataLoader_Word2Vec("train.jsonl", "Resnet152", device, embed_dir)
val_data = ImFeatureDataLoader_Word2Vec("dev.jsonl", "Resnet152", device, embed_dir, embedding_dict=train_data.embedding_dict)

def train(train_data, val_data, model, file_name, lr=1e-3, write=True):
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-2)

    trainer = Trainer(model, train_data, val_data, opt, loss, file_name=file_name, save_data=write)
    return trainer.train()

def prepare_train():
    model = None
    file_name="Word2Vec"
    if len(sys.argv) == 2:
        print("Loading pretrained model...")
        model = torch.load(sys.argv[1])
        for param in model.LSTM.parameters():
            param.requires_grad = False
        #for param in model.decoder.parameters():
        #    param.requires_grad = False
    elif len(sys.argv) == 5:
        lstm_dim = int(sys.argv[1])
        decoder_dim = int(sys.argv[2])
        dropout = float(sys.argv[3])
        file_name = sys.argv[4]
        model = LSTM_Concat(lstm_dim, 2, True, train_data.embed_dim, 
                    train_data.image_embed_dim, decoder_dim, lstm_dropout=dropout).to(device)
    else:
        model = LSTM_Concat(512, 2, True, train_data.embed_dim, train_data.image_embed_dim, 256).to(device)

    train(train_data, val_data, model, file_name)

def param_sweep():
    lstm_dims = [64, 128, 256, 512, 1024] # 5
    decoder_dims = [64, 128, 256, 512] # 4
    dropouts = [0, 0.4, 0.6] # 3 / 5 [0, 0.2, 0.4, 0.6, 0.8]
    lrs = [1e-4, 5e-4, 1e-3, 5e-3] # 4

    best_params_acc = None
    best_params_auroc = None

    best_acc = -1
    best_auroc = -1

    results = []

    for lstm_dim in lstm_dims:
        for decoder_dim in decoder_dims:
            for dropout in dropouts:
                for lr in lrs:
                    file_name = "Paramsweep_lstmconcat_" + str(lstm_dim) + "_" + str(decoder_dim) + "_" + str(dropout) + "_" + str(lr)
                    model = LSTM_Concat(lstm_dim, 2, True, train_data.embed_dim, 
                        train_data.image_embed_dim, decoder_dim, lstm_dropout=dropout).to(device)
                    acc, auroc = train(train_data, val_data, model, file_name, lr=lr, write=False)
                    if acc > best_acc:
                        best_acc = acc
                        best_params_acc = (lstm_dim, decoder_dim, dropout, lr)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_params_auroc = (lstm_dim, decoder_dim, dropout, lr)

                    results.append((lstm_dim, decoder_dim, dropout, lr, acc, auroc))
    
    for _ in range(10):
        print("~~~~~~~")

    print()
    print()
    print("Full Results: ")
    for res in results:
        print(res)
        print()

    print()
    print()
    print()

    print("Best acc:")
    print(best_acc)
    print(best_params_acc)
    print("Best auroc:")
    print(best_auroc)
    print(best_params_auroc)

def main():
    if "param_sweep" in sys.argv:
        param_sweep()
    else:
        prepare_train()

if __name__ == "__main__":
    main()