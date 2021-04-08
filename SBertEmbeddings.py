from sentence_transformers import SentenceTransformer
import sys
import numpy as np
import json
import os

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

input_text = sys.argv[1]
output_dir = sys.argv[2]

index = open(input_text, "r")

for line in index:
    entry = json.loads(line)

    text = entry["text"]

    output = model.encode([text])

    output = output[0,:].detach().numpy()

    out_path = os.path.join(output_dir, str(entry["id"]))
    np.save(out_path, output)