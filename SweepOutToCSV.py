import sys

f = open(sys.argv[1], "r")
out = open(sys.argv[2], "w")

for line in f:
    line = line.replace("(", "").replace(")", "").strip()
    if line != "":
        out.write("LSTM Concat, " + line + "\n")