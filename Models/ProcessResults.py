import sys

results_output = sys.argv[1]

f = open(results_output, "r")

max_acc = 0
epoch = -1

process_line = False
curr_epoch = 0
for line in f:
    if "Eval acc: " in line:
        process_line = True
        curr_epoch = curr_epoch + 1
    elif process_line:
        process_line = False
        result = float(line)
        if result > max_acc:
            max_acc = result
            epoch = curr_epoch

print("Max evaluation accuracy:")
print(max_acc)
print("Epoch " + str(epoch))