from transformers import BertTokenizer, BertModel
import sys
import numpy as np
import json
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

input_text = sys.argv[1]
output_dir = sys.argv[2]

index = open(input_text, "r")

for line in index:
    entry = json.loads(line)

    text = entry["text"]

    text_input_encoding = tokenizer.encode_plus(text, return_tensors = "pt")

    output = model(**text_input_encoding)

    last_hidden_state = output[0]
    pooler_output = output[1]
    pooler_output = pooler_output[0,:].detach().numpy()

    out_path = os.path.join(output_dir, str(entry[id]) + ".npz")
    np.save(out_path, pooler_output)