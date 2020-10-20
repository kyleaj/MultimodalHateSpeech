import sys

results_output = sys.argv[1]

f = open(results_output, "r")

max_acc = 0
epoch = -1

for i, line in enumerate(f):
    if "Eval acc: " in line:
        result = line.split(":")[1]
        result = float(result)
        if result > max_acc:
            max_acc = result
            epoch = i+1

print("Max evaluation accuracy:")
print(max_acc)
print("Epoch " + str(epoch))