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

class BaseImFeatureDataLoader:

    def __init__(self):
        self.ims = None
        self.captions = None
        self.labels = None
        self.order = None
        self.lengths = None
        self.device = None

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

    def get_capitalization_feature(self, word):
        upper_count = sum(1 for c in word if c.isupper())
        return upper_count/len(word)

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

class ImFeatureDataLoader(BaseImFeatureDataLoader):

    def __init__(self, path_to_json, image_network, device):
        super().__init__()

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
    def __init__(self, path_to_json, image_network, device, embeddings_path, remove_stop_words=True, embedding_dict=None, add_cap_feat=True):
        self.remove_stop_words = remove_stop_words
        self.embeddings_path = os.path.join("Embeddings", embeddings_path)
        self.embedding_dict = embedding_dict
        self.add_cap_feat = add_cap_feat
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
        self.embed_dim = len(self.embedding_dict["the"]) + (1 if self.add_cap_feat else 0)
        unknown = np.zeros((len(self.embedding_dict["the"]) + 1)) if self.add_cap_feat else np.zeros_like(self.embedding_dict["the"])
        self.lengths = [0] * len(self.captions)

        print("Embedding dimension: " + str(self.embed_dim))

        for i in range(len(self.captions)):
            caption = self.captions[i]
            one_hot_caption = [np.zeros(self.embed_dim)] * max_len
            length = 0
            for word in caption:
                word_orig = word
                word = word.lower()
                if self.remove_stop_words and word in eng_stopwords:
                    continue
                if word.isalpha() or word.replace("_", "").isalpha():
                    embed = unknown
                    if word in self.embedding_dict:
                        embed = self.embedding_dict[word]
                        if self.add_cap_feat:
                            #print(self.get_capitalization_feature(word_orig))
                            #print(embed.shape)
                            embed = list(embed) + [self.get_capitalization_feature(word_orig)]
                            #print(len(embed))
                            embed = np.array(embed)
                            #if embed.dtype != np.float:
                            #    print(embed)
                            #    print("Not float!")
                            #    print(embed.dtype)
                            #    print(word_orig)
                            #    print(len(embed))
                            #    print(self.get_capitalization_feature(word_orig))
                            #print(embed.shape)
                            #print(embed.dtype)
                            #print(type(embed))
                            #exit(0)
                    one_hot_caption[length] = embed
                length += 1

            self.lengths[i] = length
            self.captions[i] = np.array(one_hot_caption)
            #for cap in one_hot_caption:
            #    print(type(cap))
            #    print(cap.dtype)
            #assert self.captions[i].dtype == np.float

        #print(self.captions.dtype)
        self.captions = np.array(self.captions)
        self.lengths = np.array(self.lengths)
        #print(self.captions.dtype)
        print("Done!")
        #exit(0)

class ImFeatureDataLoader_Flickr(BaseImFeatureDataLoader):
    def __init__(self, path_to_csv, path_to_ims, device):
        super().__init__()

        self.ims = {}
        self.im_keys = []
        self.captions = []
        self.order = []
        self.lengths = None
        self.device = device

        index = open(path_to_csv, "r")
        print("Loading...")

        for line in index:
            pieces =  line.split("|")
            if (len(pieces) != 3):
                print("Invalid entry:")
                print(line)
                continue

            im_file, _, text = pieces

            key = im_file

            im_features = im_file + ".npy"
            im_features = os.path.join(path_to_ims, im_features)

            if key not in self.ims:
                if not os.path.exists(im_features):
                    print("Couldn't find " + str(im_features))
                    print(im_features)
                    continue
                im_features = np.load(im_features)
                self.ims[key] = im_features
                self.im_keys.append(key)

            text = word_tokenize(text)
            text = [BOS] + text + [EOS]
            self.captions.append((key, text))

        self.order = np.random.permutation(len(self.captions))

        self.image_embed_dim = self.ims[self.im_keys[0]].shape[0]
        
        print("Done!")

        self.post_process_text()

    def get_batch(self, batch_size, batch_num):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size

        indices = self.order[start:end]

        text = []
        ims = []
        labels = []
        lengths = []

        for index in indices:
            key, caption = self.captions[index]
            text.append(caption)
            ims.append(self.ims[key])
            labels.append(1)
            lengths.append(self.lengths[index])

        for index in indices:
            if np.random.randint(0, 2) == 1:
                key, caption = self.captions[index]
                text.append(caption)
                new_key = key
                new_index = -1
                while new_key != key:
                    new_index = np.random.randint(0, len(self.im_keys))
                    new_key = self.im_keys[new_index]
                
                ims.append(self.ims[new_key])
                labels.append(0)
                lengths.append(self.lengths[new_index])
            else:
                key, _ = self.captions[index]
                ims.append(self.ims[key])
                new_index = index
                while new_index != index:
                    new_index = np.random.randint(0, len(self.im_keys))
                labels.append(0)
                lengths.append(self.lengths[new_index])
                _, caption = self.captions[new_index]
                text.append(caption)


        ims = torch.Tensor(ims).to(self.device)
        text = torch.Tensor(text).to(self.device)
        labels = torch.Tensor(labels).to(self.device).long()
        lengths = torch.Tensor(lengths).to(self.device).long()

        shuffle = torch.randperm(len(indices)*2)

        ims = ims[shuffle]
        text = text[shuffle]
        labels = labels[shuffle]
        lengths = lengths[shuffle]


        return text, ims, labels, lengths

class ImFeatureDataLoader_Flickr_Word2Vec(ImFeatureDataLoader_Flickr):
    def __init__(self, path_to_csv, path_to_ims, device, embeddings_path, remove_stop_words=True, embedding_dict=None):
        self.remove_stop_words = remove_stop_words
        self.embeddings_path = os.path.join("Embeddings", embeddings_path)
        self.embedding_dict = embedding_dict
        super().__init__(path_to_csv, path_to_ims, device)

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
        for key, caption in self.captions:
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
            key, caption = self.captions[i]
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
            self.captions[i] = (key, np.array(one_hot_caption))

        print("Done!")


class ImFeatureDataLoader_Coco(BaseImFeatureDataLoader):
    def __init__(self, path_to_json, path_to_ims, device):
        super().__init__()

        self.ims = {}
        self.im_keys = []
        self.captions = []
        self.order = []
        self.lengths = None
        self.device = device

        with open(path_to_json, "r") as f:
            index = json.load(f)
            print("Loading...")

            for line in index["annotations"]:
                im_id = line["image_id"]
                text = line["caption"]

                key = im_id

                im_id_str = str(im_id)

                while len(im_id_str) < 12:
                    im_id_str = "0" + im_id_str

                im_features = "COCO_train2014_" + im_id_str + ".jpg.npy"
                im_features = os.path.join(path_to_ims, im_features)

                if key not in self.ims:
                    if not os.path.exists(im_features):
                        print("Couldn't find " + str(im_features))
                        print(im_features)
                        continue
                    im_features = np.load(im_features)
                    self.ims[key] = im_features
                    self.im_keys.append(key)

                text = word_tokenize(text)
                text = [BOS] + text + [EOS]
                self.captions.append((key, text))

        self.order = np.random.permutation(len(self.captions))

        self.image_embed_dim = self.ims[self.im_keys[0]].shape[0]
        
        print("Done!")

        self.post_process_text()

    def get_batch(self, batch_size, batch_num):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size

        indices = self.order[start:end]

        text = []
        ims = []
        labels = []
        lengths = []

        for index in indices:
            key, caption = self.captions[index]
            text.append(caption)
            ims.append(self.ims[key])
            labels.append(1)
            lengths.append(self.lengths[index])

        for index in indices:
            if np.random.randint(0, 2) == 1:
                key, caption = self.captions[index]
                text.append(caption)
                new_key = key
                new_index = -1
                while new_key != key:
                    new_index = np.random.randint(0, len(self.im_keys))
                    new_key = self.im_keys[new_index]
                
                ims.append(self.ims[new_key])
                labels.append(0)
                lengths.append(self.lengths[new_index])
            else:
                key, _ = self.captions[index]
                ims.append(self.ims[key])
                new_index = index
                while new_index != index:
                    new_index = np.random.randint(0, len(self.im_keys))
                labels.append(0)
                lengths.append(self.lengths[new_index])
                _, caption = self.captions[new_index]
                text.append(caption)


        ims = torch.Tensor(ims).to(self.device)
        text = torch.Tensor(text).to(self.device)
        labels = torch.Tensor(labels).to(self.device).long()
        lengths = torch.Tensor(lengths).to(self.device).long()

        shuffle = torch.randperm(len(indices)*2)

        ims = ims[shuffle]
        text = text[shuffle]
        labels = labels[shuffle]
        lengths = lengths[shuffle]


        return text, ims, labels, lengths

class ImFeatureDataLoader_Coco_Word2Vec(ImFeatureDataLoader_Coco):
    def __init__(self, path_to_json, path_to_ims, device, embeddings_path, remove_stop_words=True, embedding_dict=None):
        self.remove_stop_words = remove_stop_words
        self.embeddings_path = os.path.join("Embeddings", embeddings_path)
        self.embedding_dict = embedding_dict
        super().__init__(path_to_json, path_to_ims, device)

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
        for key, caption in self.captions:
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
            key, caption = self.captions[i]
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
            self.captions[i] = (key, np.array(one_hot_caption))

        print("Done!")