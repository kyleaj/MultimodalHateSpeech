import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import torch
import json
import os
import gensim

nltk.download('punkt')
nltk.download('stopwords')

BOS = "__BEGINNING_OF_SENTENCE__"
EOS = "__END_OF_SENTENCE__"
UNK = "__UNKNOWN__"

class ImFeatureDataLoader:

    def __init__(self, path_to_json, image_network, device):
        self.ims = []
        self.captions = []
        self.labels = []
        self.order = []
        self.lengths = None
        self.device = device

        index = open(path_to_json, "r")
        print("Loading...")

        for line in index:
            entry = json.loads(line)

            im_features = entry["img"] + ".npy"
            im_features = im_features.split("/")[-1]
            im_features = os.path.join("Image Features", image_network, im_features)

            if not os.path.exists(im_features):
                print("Couldn't find " + str(entry["id"]))
                print(im_features)
                continue
            im_features = np.load(im_features)
            self.ims.append(im_features)

            text = entry["text"]
            text = word_tokenize(text)
            text = [BOS] + text + [EOS]
            self.captions.append(text)

            self.labels.append(entry["label"])

        self.order = np.random.permutation(len(self.ims))

        self.image_embed_dim = self.ims[0].shape[0]
        
        print("Done!")

        self.post_process_images()
        self.post_process_labels()
        self.post_process_text()

    def post_process_images(self):
        self.ims = np.array(self.ims)

    def post_process_text(self):
        self.captions = np.array(self.captions)

    def post_process_labels(self):
        self.labels = np.array(self.labels)

    def get_batches_in_epoch(self, batch_size):
        return (len(self.order) + 1) // batch_size

    def shuffle(self):
        self.order = np.random.permutation(len(self.order))

    def get_batch(self, batch_size, batch_num):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size

        indices = self.order[start:end]

        ims = self.ims[indices]
        ims = torch.Tensor(ims).to(self.device)

        text = self.captions[indices]
        text = torch.Tensor(text).to(self.device)

        labels = self.labels[indices]
        labels = torch.Tensor(labels).to(self.device).long()

        lengths = self.lengths[indices]
        lengths = torch.Tensor(lengths).to(self.device).long()

        return text, ims, labels, lengths

    def get(self):
        return self.get_batch(len(self.order), 0)


class ImFeatureDataLoader_OneHotEmbed(ImFeatureDataLoader):
    def __init__(self, path_to_json, image_network, device, remove_stop_words=True, min_word_occurence=4, vocabulary=None, max_len=None):
        self.remove_stop_words = remove_stop_words
        self.min_word_occurence = 4
        self.vocabulary = vocabulary
        self.max_len = max_len
        super().__init__(path_to_json, image_network, device)

    def post_process_text(self):
        print("Post processing text...")

        eng_stopwords = stopwords.words('english')

        max_len = -1
        if self.vocabulary is None:
            print("Generating vocab...")
            frequency_count = {}
            for caption in self.captions:
                length = 0
                for word in caption:
                    word = word.lower()
                    if self.remove_stop_words and word in eng_stopwords:
                        continue
                    length += 1
                    if word.isalpha() or word.replace("_", "").isalpha():
                        if word in frequency_count:
                            frequency_count[word] += 1
                        else:
                            frequency_count[word] = 1
                if length > max_len:
                    max_len = length

            f = open("frequencies.txt", "w", encoding="utf-8")
            f.write(str(frequency_count))
            f.close()
            print("Finished finding frequencies")
            print("Max sentence length without stopwords: " + str(max_len))
            sorted_order = sorted(frequency_count.items(), key=lambda x: x[1] * -1)

            vocabulary = [UNK]

            for item in sorted_order:
                if (item[1] > self.min_word_occurence):
                    vocabulary.append(item[0])
            
            f = open("generated_vocabulary.txt", "w")
            f.write(str(vocabulary))
            f.close()

        else:
            print("Using already generated vocab")
            max_len = self.max_len
            vocabulary = self.vocabulary

        self.embed_dim = len(vocabulary)
        self.lengths = [0] * len(self.captions)

        print("Vocab size: " + str(self.embed_dim))

        for i in range(len(self.captions)):
            caption = self.captions[i]
            one_hot_caption = [np.zeros(len(vocabulary))] * max_len
            length = 0
            for word in caption:
                word = word.lower()
                if self.remove_stop_words and word in eng_stopwords:
                    continue
                if word.isalpha() or word.replace("_", "").isalpha():
                    embed = np.zeros((len(vocabulary)))
                    if word in vocabulary:
                        embed[vocabulary.index(word)] = 1
                    else:
                        embed[0] = 1
                    one_hot_caption[length] = embed
                length += 1

            self.lengths[i] = length
            self.captions[i] = np.array(one_hot_caption)

        self.captions = np.array(self.captions)
        self.lengths = np.array(self.lengths)
        self.vocabulary = vocabulary

        self.max_len = max_len

        print("Done!")

class ImFeatureDataLoader_Glove(ImFeatureDataLoader):
    def __init__(self, path_to_json, image_network, device, embeddings_path, remove_stop_words=True, embedding_dict=None):
        self.remove_stop_words = remove_stop_words
        self.embeddings_path = os.path.join("Embeddings", embeddings_path)
        self.embedding_dict = embedding_dict
        super().__init__(path_to_json, image_network, device)

    def post_process_text(self):
        if self.embedding_dict is None:
            print("Loading embedding dictionary...")
            self.embedding_dict = {}
            f = open(self.embeddings_path, "r")
            for line in f:
                line = line.split()
                word = line[0]
                embedding = np.array(line[1:], dtype=float)
                self.embedding_dict[word] = embedding
            print("Done!")
        else:
            print("Reusing embedding dictionary")

        eng_stopwords = stopwords.words('english')

        print("Post processing text...")

        print("Getting max text length")
        max_len = -1
        for caption in self.captions:
            length = 0
            for word in caption:
                word = word.lower()
                if self.remove_stop_words and word in eng_stopwords:
                    continue
                length += 1
            if length > max_len:
                max_len = length

        self.embed_dim = len(self.embedding_dict["the"]) # Assuming it has an embedding for "the"...
        unknown = np.zeros_like(self.embedding_dict["the"])
        self.lengths = [0] * len(self.captions)

        print("Embedding dimension: " + str(self.embed_dim))

        for i in range(len(self.captions)):
            caption = self.captions[i]
            one_hot_caption = [np.zeros(self.embed_dim)] * max_len
            length = 0
            for word in caption:
                word = word.lower()
                if self.remove_stop_words and word in eng_stopwords:
                    continue
                if word.isalpha() or word.replace("_", "").isalpha():
                    embed = unknown
                    if word in self.embedding_dict:
                        embed = self.embedding_dict[word]
                    one_hot_caption[length] = embed
                length += 1

            self.lengths[i] = length
            self.captions[i] = np.array(one_hot_caption)

        self.captions = np.array(self.captions)
        self.lengths = np.array(self.lengths)

        print("Done!")

class ImFeatureDataLoader_Word2Vec(ImFeatureDataLoader):
    def __init__(self, path_to_json, image_network, device, embeddings_path, remove_stop_words=True, embedding_dict=None):
        self.remove_stop_words = remove_stop_words
        self.embeddings_path = os.path.join("Embeddings", embeddings_path)
        self.embedding_dict = embedding_dict
        super().__init__(path_to_json, image_network, device)

    def post_process_text(self):
        if self.embedding_dict is None:
            print("Loading embeddings...")
            self.embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(self.embeddings_path, binary=True)
            print("Done!")
        else:
            print("Reusing embedding dictionary")

        eng_stopwords = stopwords.words('english')

        print("Post processing text...")

        print("Getting max text length")
        max_len = -1
        for caption in self.captions:
            length = 0
            for word in caption:
                word = word.lower()
                if self.remove_stop_words and word in eng_stopwords:
                    continue
                length += 1
            if length > max_len:
                max_len = length

        assert "the" in self.embedding_dict  # Assuming it has an embedding for "the"...
        self.embed_dim = len(self.embedding_dict["the"])
        unknown = np.zeros_like(self.embedding_dict["the"])
        self.lengths = [0] * len(self.captions)

        print("Embedding dimension: " + str(self.embed_dim))

        for i in range(len(self.captions)):
            caption = self.captions[i]
            one_hot_caption = [np.zeros(self.embed_dim)] * max_len
            length = 0
            for word in caption:
                word = word.lower()
                if self.remove_stop_words and word in eng_stopwords:
                    continue
                if word.isalpha() or word.replace("_", "").isalpha():
                    embed = unknown
                    if word in self.embedding_dict:
                        embed = self.embedding_dict[word]
                    one_hot_caption[length] = embed
                length += 1

            self.lengths[i] = length
            self.captions[i] = np.array(one_hot_caption)

        self.captions = np.array(self.captions)
        self.lengths = np.array(self.lengths)

        print("Done!")