import torch
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class MCB_RaceGender(torch.nn.Module):

    def __init__(self, LSTM_dim, LSTM_cell_num, LSTM_bidirectional, text_embed_dim, image_embed_dim, decoder_dim, num_classes=2):
        super().__init__()

        self.LSTM = torch.nn.LSTM(
            input_size=text_embed_dim, 
            hidden_size=LSTM_dim,
            num_layers=LSTM_cell_num,
            bidirectional=LSTM_bidirectional,
            batch_first=True)

        lstm_out = LSTM_dim
        if (LSTM_bidirectional):
            lstm_out *= 2

        self.mcb = CompactBilinearPooling(lstm_out, image_embed_dim, decoder_dim)

        self.decoder = torch.nn.Linear(in_features=decoder_dim + 9, out_features=decoder_dim)
        self.classifier = torch.nn.Linear(in_features=decoder_dim, out_features=num_classes)

    def forward(self, text, image, race_lengths):
        race_gender_embedding, lengths = race_lengths
        input = torch.nn.utils.rnn.pack_padded_sequence(text, lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.LSTM(input)

        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        idx = (lengths.long() - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
        idx = idx.unsqueeze(1)
        # Shape: (batch_size, rnn_hidden_dim)
        lstm_out = lstm_out.gather(1, idx).squeeze(1)

        mcb_out = self.mcb(lstm_out, image)

        mcb_out = torch.cat((mcb_out, race_gender_embedding), dim=1)

        decoder_out = self.decoder(mcb_out)
        decoder_out = torch.nn.ReLU()(decoder_out)

        out = self.classifier(decoder_out)

        return out