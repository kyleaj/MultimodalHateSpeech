import torch
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class MCB_Late_Fusion(torch.nn.Module):

    def __init__(self, text_embed_dim, image_embed_dim, decoder_dim, num_classes=2):
        super().__init__()

        self.mcb = CompactBilinearPooling(text_embed_dim, image_embed_dim, decoder_dim)

        self.decoder = torch.nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.classifier = torch.nn.Linear(in_features=decoder_dim, out_features=num_classes)

    def forward(self, text, image, lengths):
        mcb_out = self.mcb(text, image)

        decoder_out = self.decoder(mcb_out)
        decoder_out = torch.nn.ReLU()(decoder_out)

        out = self.classifier(decoder_out)

        return out