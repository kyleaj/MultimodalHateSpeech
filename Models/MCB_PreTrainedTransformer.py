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

        #self.drop1 = torch.nn.Dropout(0.5)
        #self.drop2 = torch.nn.Dropout(0.5)
        #self.drop3 = torch.nn.Dropout(0.2)

    def forward(self, text, image, lengths):
        text = self.text_process(text)
        #text = self.drop1(text)
        text = torch.nn.ReLU()(text)

        image = self.im_process(image)
        #image = self.drop2(image)
        image = torch.nn.ReLU()(image)

        mcb_out = self.mcb(text, image)

        decoder_out = self.decoder(mcb_out)
        #decoder_out = self.drop3(decoder_out)
        decoder_out = torch.nn.ReLU()(decoder_out)

        out = self.classifier(decoder_out)

        return out