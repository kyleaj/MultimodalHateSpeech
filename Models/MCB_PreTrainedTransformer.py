import torch
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class MCB_Late_Fusion(torch.nn.Module):

    def __init__(self, text_embed_dim, image_embed_dim, decoder_dim, num_classes=2):
        super().__init__()

        self.im_process = torch.nn.Linear(in_features=image_embed_dim, out_features=512)
        self.text_process = torch.nn.Linear(in_features=text_embed_dim, out_features=512)

        self.mcb = CompactBilinearPooling(512, 512, 512)

        self.decoder = torch.nn.Linear(in_features=512, out_features=256)
        self.classifier = torch.nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, text, image, lengths):
        text = self.text_process(text)
        text = torch.nn.Dropout(0.5)(text)
        text = torch.nn.ReLU()(text)

        image = self.im_process(image)
        image = torch.nn.Dropout(0.5)(image)
        image = torch.nn.ReLU()(image)

        mcb_out = self.mcb(text, image)

        decoder_out = self.decoder(mcb_out)
        decoder_out = torch.nn.Dropout(0.5)(decoder_out)
        decoder_out = torch.nn.ReLU()(decoder_out)

        out = self.classifier(decoder_out)

        return out