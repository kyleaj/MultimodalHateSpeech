import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import time
import sys
from PIL import Image
import os

image_dir = sys.argv[1]
out_dir = sys.argv[2]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU!")
else:
    device = torch.device("cpu")
    print("Using CPU :(")

# models = ["resnet", "alexnet", "vgg", 
# "squeezenet", "densenet", "inception"]

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
model.fc = nn.Identity()
model.eval()

model.to(device)

start_time = time.time()

ims = os.listdir(image_dir)
im_num = len(ims)

failed = []

remaining = -1

for i, im in enumerate(ims):
    outpath = os.path.join(out_dir, im, ".npy")
    if (os.path.exists(outpath)) or not("jpg" in im.lower() or "png" in im.lower() or "jpeg" in im.lower()):
        continue

    filename = os.path.join(image_dir, im)
    input_image = Image.open(filename)

    try:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)

        output = output[0]
        output = output.numpy()
        
        np.save(outpath, output)

        elapsed = time.time() - start_time
        speed = (i+1) / elapsed
        remaining = (im_num - i - 1) / speed
        remaining = int(remaining * 100) / 100.0
    except:
        failed.append(im)

    print(str(i+1) + " / " + str(im_num) + ", about " + str(remaining) + "s left.")
    sys.stdout.write("\033[F") # Cursor up one line

print("Num failed: " + str(len(failed)))
f = open(os.path.join(out_dir, "failures"), "w")
for fail in failed:
    f.write(str(fail))
f.close()