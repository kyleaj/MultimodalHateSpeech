import pickle
import sys

d = pickle.load(open(sys.argv[1],"rb"), encoding="latin1")

dic = {}

for word, vect in zip(d[0], d[1]):
    dic[word] = vect

pickle.dump(dic, open(sys.argv[1], "wb"), protocol=pickle.HIGHEST_PROTOCOL)