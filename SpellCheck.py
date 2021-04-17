from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
import numpy as np
import json
import os
from nltk.tokenize import word_tokenize
import gensim
import torch
import nltk
import json

class SpellChecker:

    def __init__(self, embeddings_path="/tigress/kyleaj/Thesis/Embeddings/GoogleNews-vectors-negative300.bin", edit_weight=2, max_edit_distance=5):
        print("Loading model...")
        self.LM_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.LM = AutoModelForMaskedLM.from_pretrained('roberta-base')
        self.MASK =self.LM_tokenizer.mask_token

        # base on word2vec
        print("Loading embedding dict...")
        self.embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

        self.ew = edit_weight
        self.em = max_edit_distance

    def eval_phrase(self, text):
        model_tokens = word_tokenize(text)
        #print(model_tokens)
        corrections = {}
        min_start_index = 0
        for word in model_tokens:
            min_start_index += text[min_start_index:].find(word, 0)
            if word.isalpha() and (not (word in self.embedding_dict)):
                #print(word)
                # probably a mispelled word!
                count = 0
                if word in corrections:
                    count = corrections[word][1] + 1
                    corrections[word][1] = count
                else:
                    corrections[word] = [[], count]
                
                start_index = text[min_start_index:].find(word, 0)
                start_index += min_start_index

                masked_text = text[:start_index] + self.MASK + text[start_index + len(word):]
                #print(masked_text)
                
                input = self.LM_tokenizer.encode(masked_text, return_tensors="pt")
                mask_token_index = torch.where(input == self.LM_tokenizer.mask_token_id)[1]

                token_logits = self.LM(input).logits

                mask_token_logits = token_logits[0, mask_token_index, :]

                mask_token_logits = mask_token_logits[0,:]
                # convert to probabilities
                mask_token_logits = torch.nn.functional.softmax(mask_token_logits)

                best_prob = -1
                best_correction = word # No corrections somehow? Default to no change.

                for i in range(len(mask_token_logits)):
                    replacement = self.LM_tokenizer.decode([i])
                    edit_distance = nltk.edit_distance(replacement, word)
                    edit_distance_prob = pow((1 / (1 + edit_distance)), self.ew)
                    if (edit_distance > self.em):
                        edit_distance_prob = 0
                    if (edit_distance == 0):
                        edit_distance_prob = 100000000000

                    LM_prob = mask_token_logits[i]

                    prob = edit_distance_prob * LM_prob
                    '''
                    if replacement == "alert" or replacement == "management":
                        print(replacement)
                        print(edit_distance)
                        print(edit_distance_prob)
                        print(LM_prob)
                        '''

                    if (prob > best_prob):
                        best_prob = prob
                        best_correction = replacement

                #print(masked_text.replace(self.MASK, best_correction))
                text = text[:start_index] + best_correction.strip() + text[start_index + len(word):]
                word = best_correction
                #print()
            min_start_index += len(word)
        return text



def main():
    checker = SpellChecker()
    print(checker.eval_phrase("This is a test of the emergency alret system"))

def runOnData(in_path, out_path):
    f_in = open(in_path, "r")
    f_out = open(out_path, "w")
    
    checker = SpellChecker()

    for line in f_in:
        entry = json.loads(line)
        text = entry["text"]
        result = checker.eval_phrase(text)
        if result != text:
            print(text + " -> " + result)
        entry["text"] = result
        f_out.write(json.dumps(entry))
        f_out.write("\n")

    f_out.close()
    f_in.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        runOnData(sys.argv[1], sys.argv[2])

