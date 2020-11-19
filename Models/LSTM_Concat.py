import torch

class LSTM_Concat(torch.nn.Module):

    def __init__(self, LSTM_dim, LSTM_cell_num, LSTM_bidirectional, 
                    text_embed_dim, image_embed_dim, decoder_dim, num_classes=2, lstm_dropout=0, decoder_dropout=0):
        super().__init__()

        self.LSTM = torch.nn.LSTM(
            input_size=text_embed_dim + image_embed_dim, 
            hidden_size=LSTM_dim,
            num_layers=LSTM_cell_num,
            bidirectional=LSTM_bidirectional,
            batch_first=True,
            dropout=lstm_dropout)

        lstm_out = LSTM_dim
        if (LSTM_bidirectional):
            lstm_out *= 2

        self.decoder = torch.nn.Linear(in_features=lstm_out, out_features=decoder_dim)
        self.decoder = torch.nn.Dropout(p=decoder_dropout)(self.decoder)
        self.classifier = torch.nn.Linear(in_features=decoder_dim, out_features=num_classes)

    def forward(self, text, image, lengths):
        image = image.view(image.shape[0], 1, image.shape[1])
        image = image.expand(image.shape[0], text.shape[1], image.shape[2])

        input = torch.cat((text, image), dim=2)

        input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.LSTM(input)

        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        idx = (lengths.long() - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
        idx = idx.unsqueeze(1)
        # Shape: (batch_size, rnn_hidden_dim)
        lstm_out = lstm_out.gather(1, idx).squeeze(1)

        decoder_out = self.decoder(lstm_out)
        decoder_out = torch.nn.ReLU()(decoder_out)

        out = self.classifier(decoder_out)

        return out